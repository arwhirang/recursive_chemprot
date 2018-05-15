"""Implementation of SPINN in TensorFlow eager execution.

SPINN: Stack-Augmented Parser-Interpreter Neural Network.

Ths file contains model definition and code for training the model.

The model definition is based on PyTorch implementation at:
  https://github.com/jekbradbury/examples/tree/spinn/snli

which was released under a BSD 3-Clause License at:
https://github.com/jekbradbury/examples/blob/spinn/LICENSE:

Copyright (c) 2017,
All rights reserved.

We modified this SPINN code a little to experiment the CHEMPROT task in Biocreative VI challenge.

Original model aimed at solving the SNLI (Standford Natural Language Inference) task, using the SPINN model from above. We changed the original model to serve chemprot data.
For details of the original task, see: https://nlp.stanford.edu/projects/snli/
For details of the chemprot task, see: http://www.biocreative.org/news/biocreative-vi/track-5/

Instructions for use:
* See `README.md` for details on how to prepare the data and word embedding vector.

References:
* Bowman, S.R., Gauthier, J., Rastogi A., Gupta, R., Manning, C.D., & Potts, C.
  (2016). A Fast Unified Model for Parsing and Sentence Understanding.
  https://arxiv.org/abs/1603.06021
* Bradbury, J. (2017). Recursive Neural Networks with PyTorch.
  https://devblogs.nvidia.com/parallelforall/recursive-neural-networks-pytorch/
* original code for this spinn:
  https://github.com/tensorflow/tensorflow/blob/master/third_party/examples/eager/spinn/spinn.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os
import sys
import time
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import tensorflow.contrib.eager as tfe
import data_chemprot
from sklearn import metrics
import pickle

def _bundle(lstm_iter):
  """Concatenate a list of Tensors along 1st axis and split result into two.

  Args:
    lstm_iter: A `list` of `N` dense `Tensor`s, each of which has the shape
      (R, 2 * M).

  Returns:
    A `list` of two dense `Tensor`s, each of which has the shape (N * R, M).
  """
  return tf.split(tf.concat(lstm_iter, 0), 2, axis=1)


def _unbundle(state):
  """Concatenate a list of Tensors along 2nd axis and split result.

  This is the inverse of `_bundle`.

  Args:
    state: A `list` of two dense `Tensor`s, each of which has the shape (R, M).

  Returns:
    A `list` of `R` dense `Tensors`, each of which has the shape (1, 2 * M).
  """
  # print(state[0].shape)
  return tf.split(tf.concat(state, 1), state[0].shape[0], axis=0)


class Reducer(tfe.Network):
  """A module that applies reduce operation on left and right vectors."""

  def __init__(self, size, tracker_size=None):
    super(Reducer, self).__init__()
    self.left = self.track_layer(tf.layers.Dense(5 * size, activation=None))
    self.right = self.track_layer(
        tf.layers.Dense(5 * size, activation=None, use_bias=False))
    if tracker_size is not None:
      self.track = self.track_layer(
          tf.layers.Dense(5 * size, activation=None, use_bias=False))
    else:
      self.track = None

  def call(self, left_in, right_in, tracking=None):
    """Invoke forward pass of the Reduce module.

    This method feeds a linear combination of `left_in`, `right_in` and
    `tracking` into a Tree LSTM and returns the output of the Tree LSTM.

    Args:
      left_in: A list of length L. Each item is a dense `Tensor` with
        the shape (1, n_dims). n_dims is the size of the embedding vector.
      right_in: A list of the same length as `left_in`. Each item should have
        the same shape as the items of `left_in`.
      tracking: Optional list of the same length as `left_in`. Each item is a
        dense `Tensor` with shape (1, tracker_size * 2). tracker_size is the
        size of the Tracker's state vector.

    Returns:
      Output: A list of length batch_size. Each item has the shape (1, n_dims).
    """
    #why do we need tracking? ==> see the paper sec 3.3

    left, right = _bundle(left_in), _bundle(right_in)
    #check code
    #print(len(left_in), left_in[0].get_shape(), len(right_in), right_in[0].get_shape())
    #print(len(left), left[0].get_shape(), len(right), right[0].get_shape())

    lstm_in = self.left(left[0]) + self.right(right[0])
    if self.track and tracking:
      lstm_in += self.track(_bundle(tracking)[0])
    return _unbundle(self._tree_lstm(left[1], right[1], lstm_in))

  def _tree_lstm(self, c1, c2, lstm_in):
    a, i, f1, f2, o = tf.split(lstm_in, 5, axis=1)
    #check code
    #print(a.get_shape(), c1.get_shape(), c2.get_shape(), lstm_in.get_shape())

    c = tf.tanh(a) * tf.sigmoid(i) + tf.sigmoid(f1) * c1 + tf.sigmoid(f2) * c2
    h = tf.sigmoid(o) * tf.tanh(c)
    return h, c


class Tracker(tfe.Network):
  """A module that tracks the history of the sentence with an LSTM."""

  def __init__(self, tracker_size, predict):
    """Constructor of Tracker.

    Args:
      tracker_size: Number of dimensions of the underlying `LSTMCell`.
      predict: (`bool`) Whether prediction mode is enabled.
    """
    super(Tracker, self).__init__()
    self._rnn = self.track_layer(tf.nn.rnn_cell.LSTMCell(tracker_size))
    self._state_size = tracker_size
    if predict:
      self._transition = self.track_layer(tf.layers.Dense(6))
    else:
      self._transition = None

  def reset_state(self):
    self.state = None

  def call(self, bufs, stacks):
    """Invoke the forward pass of the Tracker module.

    This method feeds the concatenation of the top two elements of the stacks
    into an LSTM cell and returns the resultant state of the LSTM cell.

    Args:
      bufs: A `list` of length batch_size. Each item is a `list` of
        max_sequence_len (maximum sequence length of the batch). Each item
        of the nested list is a dense `Tensor` of shape (1, d_proj), where
        d_proj is the size of the word embedding vector or the size of the
        vector space that the word embedding vector is projected to.
      stacks: A `list` of size batch_size. Each item is a `list` of
        variable length corresponding to the current height of the stack.
        Each item of the nested list is a dense `Tensor` of shape (1, d_proj).

    Returns:
      1. A list of length batch_size. Each item is a dense `Tensor` of shape
        (1, d_tracker * 2).
      2.  If under predict mode, result of applying a Dense layer on the
        first state vector of the RNN. Else, `None`.
    """
    buf = _bundle([buf[-1] for buf in bufs])[0]
    stack1 = _bundle([stack[-1] for stack in stacks])[0]
    stack2 = _bundle([stack[-2] for stack in stacks])[0]
    x = tf.concat([buf, stack1, stack2], 1)
    if self.state is None:
      batch_size = int(x.shape[0])
      zeros = tf.zeros((batch_size, self._state_size), dtype=tf.float32)
      self.state = [zeros, zeros]
    _, self.state = self._rnn(x, self.state)
    if self._transition:
      return _unbundle(self.state), self._transition(self.state[0])
    else:
      return _unbundle(self.state), None


class SPINN(tfe.Network):
  """Stack-augmented Parser-Interpreter Neural Network.

  See https://arxiv.org/abs/1603.06021 for more details.
  """

  def __init__(self, config):
    """Constructor of SPINN.

    Args:
      config: A `namedtupled` with the following attributes.
        d_proj - (`int`) number of dimensions of the vector space to project the
          word embeddings to.
        d_tracker - (`int`) number of dimensions of the Tracker's state vector.
        d_hidden - (`int`) number of the dimensions of the hidden state, for the
          Reducer module.
        n_mlp_layers - (`int`) number of multi-layer perceptron layers to use to
          convert the output of the `Feature` module to logits.
        predict - (`bool`) Whether the Tracker will enabled predictions.
    """
    super(SPINN, self).__init__()
    self.config = config
    self.reducer = self.track_layer(Reducer(config.d_hidden, config.d_tracker))
    if config.d_tracker is not None:#default = 64
      self.tracker = self.track_layer(Tracker(config.d_tracker, config.predict))
    else:
      self.tracker = None

  def call(self, buffers, transitions, training=False):
    """Invoke the forward pass of the SPINN model.

    Args:
      buffers: Dense `Tensor` of shape
        (max_sequence_len, batch_size, config.d_proj).
      transitions: Dense `Tensor` with integer values that represent the parse
        trees of the sentences. A value of 2 indicates "reduce"; a value of 3
        indicates "shift". Shape: (max_sequence_len * 2 - 3, batch_size).
      training: Whether the invocation is under training mode.

    Returns:
      Output `Tensor` of shape (batch_size, config.d_embed).
    """
    max_sequence_len, batch_size, d_proj = (int(x) for x in buffers.shape)

    # Split the buffers into left and right word items and put the initial
    # items in a stack.
    splitted = tf.split(
        tf.reshape(tf.transpose(buffers, [1, 0, 2]), [-1, d_proj]),
        max_sequence_len * batch_size, axis=0)
    buffers = [splitted[k:k + max_sequence_len]
               for k in xrange(0, len(splitted), max_sequence_len)]
    stacks = [[buf[0], buf[0]] for buf in buffers]

    if self.tracker:
      # Reset tracker state for new batch.
      self.tracker.reset_state()

    num_transitions = transitions.shape[0]

    # Iterate through transitions and perform the appropriate stack-pop, reduce
    # and stack-push operations.
    transitions = transitions.numpy()
    for i in xrange(num_transitions):
      trans = transitions[i]
      if self.tracker:
        # Invoke tracker to obtain the current tracker states for the sentences.
        tracker_states, trans_hypothesis = self.tracker(buffers, stacks)
        if trans_hypothesis:
          trans = tf.argmax(trans_hypothesis, axis=-1)
      else:
        tracker_states = itertools.repeat(None)
      lefts, rights, trackings = [], [], []
      for transition, buf, stack, tracking in zip(
          trans, buffers, stacks, tracker_states):
        if int(transition) == 3:  # Shift.
          stack.append(buf.pop())
        elif int(transition) == 2:  # Reduce.
          rights.append(stack.pop())
          lefts.append(stack.pop())
          trackings.append(tracking)

      if rights:
        #check code
        #print(len(lefts), lefts[0].get_shape(), len(rights), rights[0].get_shape(), len(trackings), trackings[0].get_shape())
        reducer_output = self.reducer(lefts, rights, trackings)
        reduced = iter(reducer_output)

        for transition, stack in zip(trans, stacks):
          if int(transition) == 2:  # Reduce.
            stack.append(next(reduced))
    return _bundle([stack.pop() for stack in stacks])[0]


class ChemprotClassifier(tfe.Network):
  """Chemprot Classifier Model.

  Original model aimed at solving the SNLI (Standford Natural Language Inference)
  task, using the SPINN model from above. We changed the original model to serve
  chemprot data.
   For details of the original task, see: https://nlp.stanford.edu/projects/snli/
   For details of the chemprot task, see: http://www.biocreative.org/news/biocreative-vi/track-5/
  """

  def __init__(self, config, embed):
    """Constructor of ChemprotClassifier.

    Args:
      config: A namedtuple containing required configurations for the model. It
        needs to have the following attributes.
        projection - (`bool`) whether the word vectors are to be projected onto
          another vector space (of `d_proj` dimensions).
        d_proj - (`int`) number of dimensions of the vector space to project the
          word embeddings to.
        embed_dropout - (`float`) dropout rate for the word embedding vectors.
        n_mlp_layers - (`int`) number of multi-layer perceptron (MLP) layers to
          use to convert the output of the `Feature` module to logits.
        mlp_dropout - (`float`) dropout rate of the MLP layers.
        d_out - (`int`) number of dimensions of the final output of the MLP
          layers.
        lr - (`float`) learning rate.
      embed: A embedding matrix of shape (vocab_size, d_embed).
    """
    super(ChemprotClassifier, self).__init__()
    self.config = config
    self.embed = tf.constant(embed)

    self.projection = self.track_layer(tf.layers.Dense(config.d_proj))#d_proj=400
    self.embed_bn = self.track_layer(tf.layers.BatchNormalization())
    self.embed_dropout = self.track_layer(
        tf.layers.Dropout(rate=config.embed_dropout))#embed_dropout=0.08
    self.encoder = self.track_layer(SPINN(config))

    self.feature_bn = self.track_layer(tf.layers.BatchNormalization())
    self.feature_dropout = self.track_layer(
        tf.layers.Dropout(rate=config.mlp_dropout))

    self.mlp_dense = []
    self.mlp_bn = []
    self.mlp_dropout = []
    for _ in xrange(config.n_mlp_layers):
      self.mlp_dense.append(self.track_layer(tf.layers.Dense(config.d_mlp)))
      self.mlp_bn.append(
          self.track_layer(tf.layers.BatchNormalization()))
      self.mlp_dropout.append(
          self.track_layer(tf.layers.Dropout(rate=config.mlp_dropout)))
    self.mlp_output = self.track_layer(tf.layers.Dense(
        config.d_out,
        kernel_initializer=tf.random_uniform_initializer(minval=-5e-3,
                                                         maxval=5e-3)))

  def call(self,
           word_ids,
           transition,
           training=False):
    """Invoke the forward pass the ChemprotClassifier model.

    Args:
      word_ids: The word indices of the sentences, with shape
        (max_seq_len, batch_size).
      transition: The transitions for the premise sentences, with shape
        (max_seq_len * 2 - 3, batch_size).
      training: Whether the invocation is under training mode.

    Returns:
      The logits, as a dense `Tensor` of shape (batch_size, d_out), where d_out
      is the size of the output vector.
    """
    # Perform embedding lookup on the inputs, which have the word-index format.
    sen_embed = tf.nn.embedding_lookup(self.embed, word_ids)#shape (36, 128, 200)
    if self.config.projection:
      # Project the embedding vectors to another vector space.
      sen_embed = self.projection(sen_embed)#dense layer

    # dropout
    sen_embed = self.embed_dropout(sen_embed, training=training)

    # Run the batch-normalized and dropout-processed word vectors through the SPINN encoder.
    spinn_logits = self.encoder(sen_embed, transition,training=training)

    # Combine encoder outputs for premises and hypotheses into logits.
    # apply dropout
    logits = self.feature_dropout(spinn_logits, training=training)

    # Apply the multi-layer perceptron on the logits.
    for dense, bn, dropout in zip(
        self.mlp_dense, self.mlp_bn, self.mlp_dropout):
      logits = tf.nn.elu(dense(logits))
      logits = dropout(bn(logits, training=training), training=training)
    logits = self.mlp_output(logits)#dense
    return logits

def _evaluate_on_dataset(chemprot_data, batch_size, model, use_gpu):
  """Run evaluation on a dataset.

  Args:
    chemprot_data: The `data_chemprot.ChemprotData` to use in this evaluation.
    batch_size: The batch size to use during this evaluation.
    model: An instance of `ChemprotClassifier` to evaluate.
    trainer: An instance of `ChemprotClassifierTrainer to use for this
      evaluation.
    use_gpu: Whether GPU is being used.

  Returns:
    1. Average loss across all examples of the dataset.
    2. Average accuracy rate across all examples of the dataset.
    3. f1 score (not the micro-averaged f1 score, just rough reference)
    4. labels        - for eval.sh
    5. logits for ensemble
    6. pmids         - for eval.sh
    7. first entity  - for eval.sh
    8. second entity - for eval.sh
  """
  mean_loss = tfe.metrics.Mean()
  accuracy = tfe.metrics.Accuracy()
  test_labels_whole = []
  test_logits_whole = []
  _pmids = []
  _ent1s = []
  _ent2s = []
  for label, word_ids, trans, pmids, ent1s, ent2s in _get_dataset_iterator_noShuffle(
          chemprot_data, batch_size):
    if use_gpu:
      label, word_ids, trans = label.gpu(), word_ids.gpu(), trans.gpu()
    # pmids, ent1s, ent2s = pmids.cpu(), ent1s.cpu(), ent2s.cpu()

    logits = model(word_ids, trans, training=False)#not trained
    loss_val = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label, logits=logits))
    batch_size = tf.shape(label)[0]
    mean_loss(loss_val, weights=batch_size.gpu() if use_gpu else batch_size)
    accuracy(tf.argmax(logits, axis=1), label)
    test_labels_whole = test_labels_whole + label.numpy().tolist()
    test_logits_whole = test_logits_whole + logits.numpy().tolist()
    _pmids = _pmids + pmids.numpy().tolist()
    _ent1s = _ent1s + ent1s.numpy().tolist()
    _ent2s = _ent2s + ent2s.numpy().tolist()
  f1score = metrics.f1_score(test_labels_whole, tf.cast(tf.argmax(test_logits_whole, 1), tf.int32), average=None)
  return mean_loss.result().numpy(), accuracy.result().numpy(), f1score, test_labels_whole, test_logits_whole, _pmids, _ent1s, _ent2s

def _get_dataset_iterator(chemprot_data, batch_size):
  """Get a data iterator for a split of Chemprot data.

  Args:
    chemprot_data: A `data_chemprot.ChemprotData` object.
    batch_size: The desired batch size.

  Returns:
    A dataset iterator.
  """
  with tf.device("/device:CPU:0"):
    # Some tf.data ops, such as ShuffleDataset, are available only on CPU.
    dataset = tf.data.Dataset.from_generator(
        chemprot_data.get_generator(batch_size),
        (tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64))
    dataset = dataset.shuffle(chemprot_data.num_batches(batch_size))

    return tfe.Iterator(dataset)

#no shuffle version is required for ensemble
def _get_dataset_iterator_noShuffle(chemprot_data, batch_size):
  """Get a data iterator for a split of Chemprot data.

  Args:
    chemprot_data: A `data_chemprot.ChemprotData` object.
    batch_size: The desired batch size.

  Returns:
    A dataset iterator.
  """
  with tf.device("/device:CPU:0"):
    # Some tf.data ops, such as ShuffleDataset, are available only on CPU.
    dataset = tf.data.Dataset.from_generator(
        chemprot_data.get_generator(batch_size),
        (tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64))

    return tfe.Iterator(dataset)

def test_spinn(embed, test_data, config):
  """Test a SPINN model.

  Args:
    embed: The embedding matrix as a float32 numpy array with shape
      [vocabulary_size, word_vector_len]. word_vector_len is the length of a
      word embedding vector.
    test_data: An instance of `data_chemprot.ChemprotData`, for the test split.
    config: A configuration object. See the argument to this Python binary for
      details.

  Returns:
    1. Final loss value on the test split.
    2. Final fraction of correct classifications on the test split.
  """
  use_gpu = tfe.num_gpus() > 0 and not config.force_cpu
  device = "gpu:0" if use_gpu else "cpu:0"
  print("Using device: %s" % device)

  log_header = (
      "  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss"
      "     Accuracy  Dev/Accuracy")
  dev_log_template = (
      "{:>6.0f} {:>5.0f} {:>9.0f} {:>5.0f}/{:<5.0f} {:>7.0f}% {:>8.6f} "
      "{:8.6f} {:12.4f} {:12.4f}")

  summary_writer = tf.contrib.summary.create_file_writer(
      config.logdir, flush_millis=10000)
  with tf.device(device), \
       summary_writer.as_default(), \
       tf.contrib.summary.always_record_summaries():
    model = ChemprotClassifier(config, embed)
    latest_checkpoint = tf.train.latest_checkpoint(config.logdir)
    print("Latest checkpoint", latest_checkpoint)
    tfe.restore_network_checkpoint(model, tf.train.latest_checkpoint(config.logdir))

    start = time.time()
    dev_mean_loss = tfe.metrics.Mean()
    dev_accuracy = tfe.metrics.Accuracy()
    print(log_header)

    #restore
    dev_loss, dev_frac_correct, dev_f1, dev_lables, dev_logits, dev_pmids, dev_ent1s, dev_ent2s = _evaluate_on_dataset(
        dev_data, config.batch_size, model, use_gpu)

    print(dev_log_template.format(
        time.time() - start,
        0, 0, 1, 0,
        1 / 1,
        0, dev_loss,
        0, dev_frac_correct * 100.0))
    print(dev_f1)

def train_spinn(embed, train_data, dev_data, config):
  """Train a SPINN model.

  Args:
    embed: The embedding matrix as a float32 numpy array with shape
      [vocabulary_size, word_vector_len]. word_vector_len is the length of a
      word embedding vector.
    train_data: An instance of `data_chemprot.ChemprotData`, for the train split.
    dev_data: Same as above, for the dev split.
    test_data: Same as above, for the test split.
    config: A configuration object. See the argument to this Python binary for
      details.

  Returns:
    1. Final loss value on the test split.
    2. Final fraction of correct classifications on the test split.
  """
  use_gpu = tfe.num_gpus() > 0 and not config.force_cpu
  device = "gpu:0" if use_gpu else "cpu:0"
  print("Using device: %s" % device)

  log_header = (
      "  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss"
      "     Accuracy  Dev/Accuracy")
  log_template = (
      "{:>6.0f} {:>5.0f} {:>9.0f} {:>5.0f}/{:<5.0f} {:>7.0f}% {:>8.6f} {} "
      "{:12.4f} {}")
  dev_log_template = (
      "{:>6.0f} {:>5.0f} {:>9.0f} {:>5.0f}/{:<5.0f} {:>7.0f}% {:>8.6f} "
      "{:8.6f} {:12.4f} {:12.4f}")

  summary_writer = tf.contrib.summary.create_file_writer(
      config.logdir, flush_millis=10000)
  train_len = train_data.num_batches(config.batch_size)
  with tf.device(device), \
       summary_writer.as_default(), \
       tf.contrib.summary.always_record_summaries():
    model = ChemprotClassifier(config, embed)
    print(config.logdir)
    global_step = tf.train.get_or_create_global_step()
    _learning_rate = tfe.Variable(config.lr, name="learning_rate")
    _optimizer = tf.train.AdamOptimizer(_learning_rate, epsilon=1e-6)

    start = time.time()
    iterations = 0
    mean_loss = tfe.metrics.Mean()
    accuracy = tfe.metrics.Accuracy()
    print(log_header)
    for epoch in xrange(config.epochs):
      batch_idx = 0
      for label, word_ids, trans, pmids, ent1s, ent2s in _get_dataset_iterator(
          train_data, config.batch_size):
        if use_gpu:
          label, word_ids = label.gpu(), word_ids.gpu()
          # trans are used for dynamic control flow

        iterations += 1
        with tfe.GradientTape() as tape:
            tape.watch(model.variables)
            logits = model(word_ids, trans, training=True)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label, logits=logits))
        gradients = tape.gradient(loss, model.variables)
        _optimizer.apply_gradients(zip(gradients, model.variables),
                                        global_step=global_step)
        batch_size = tf.shape(label)[0]
        mean_loss(loss.numpy(),
                  weights=batch_size.gpu() if use_gpu else batch_size)
        accuracy(tf.argmax(logits, axis=1), label)

        if iterations % config.save_every == 0:
          #tfe.Saver(model.variables).save(os.path.join(config.logdir, "ckpt"), global_step=global_step)
          # This line works.
          tfe.save_network_checkpoint(model, os.path.join(config.logdir, "ckpt"), global_step=global_step)

        if iterations % config.dev_every == 0:
          dev_loss, dev_frac_correct, dev_f1, dev_lables, dev_logits, dev_pmids, dev_ent1s, dev_ent2s = _evaluate_on_dataset(
              dev_data, config.batch_size, model, use_gpu)
          print(dev_log_template.format(
              time.time() - start,
              epoch, iterations, 1 + batch_idx, train_len,
              100.0 * (1 + batch_idx) / train_len,
              mean_loss.result(), dev_loss,
              accuracy.result() * 100.0, dev_frac_correct * 100.0))
          print(dev_f1)#(not the micro-averaged f1 score, just rough reference)

          fgold = open("gold", "w")
          for i in range(len(dev_pmids)):
              if dev_lables[i] == 0:
                  continue
              elif dev_lables[i] == 1:
                  fgold.write(str(dev_pmids[i]))
                  fgold.write("\tCPR:3\tArg1:T")
                  fgold.write(str(dev_ent1s[i]))
                  fgold.write("\tArg2:T")
                  fgold.write(str(dev_ent2s[i]))
                  fgold.write("\n")
              elif dev_lables[i] == 2:
                  fgold.write(str(dev_pmids[i]))
                  fgold.write("\tCPR:4\tArg1:T")
                  fgold.write(str(dev_ent1s[i]))
                  fgold.write("\tArg2:T")
                  fgold.write(str(dev_ent2s[i]))
                  fgold.write("\n")
              elif dev_lables[i] == 3:
                  fgold.write(str(dev_pmids[i]))
                  fgold.write("\tCPR:5\tArg1:T")
                  fgold.write(str(dev_ent1s[i]))
                  fgold.write("\tArg2:T")
                  fgold.write(str(dev_ent2s[i]))
                  fgold.write("\n")
              elif dev_lables[i] == 4:
                  fgold.write(str(dev_pmids[i]))
                  fgold.write("\tCPR:6\tArg1:T")
                  fgold.write(str(dev_ent1s[i]))
                  fgold.write("\tArg2:T")
                  fgold.write(str(dev_ent2s[i]))
                  fgold.write("\n")
              elif dev_lables[i] == 5:
                  fgold.write(str(dev_pmids[i]))
                  fgold.write("\tCPR:9\tArg1:T")
                  fgold.write(str(dev_ent1s[i]))
                  fgold.write("\tArg2:T")
                  fgold.write(str(dev_ent2s[i]))
                  fgold.write("\n")
          fgold.close()
          dev_preds = np.argmax(dev_logits, 1)
          fpred = open("pred", "w")
          for i in range(len(dev_pmids)):
              if dev_preds[i] == 0:
                  continue
              elif dev_preds[i] == 1:
                  fpred.write(str(dev_pmids[i]))
                  fpred.write("\tCPR:3\tArg1:T")
                  fpred.write(str(dev_ent1s[i]))
                  fpred.write("\tArg2:T")
                  fpred.write(str(dev_ent2s[i]))
                  fpred.write("\n")
              elif dev_preds[i] == 2:
                  fpred.write(str(dev_pmids[i]))
                  fpred.write("\tCPR:4\tArg1:T")
                  fpred.write(str(dev_ent1s[i]))
                  fpred.write("\tArg2:T")
                  fpred.write(str(dev_ent2s[i]))
                  fpred.write("\n")
              elif dev_preds[i] == 3:
                  fpred.write(str(dev_pmids[i]))
                  fpred.write("\tCPR:5\tArg1:T")
                  fpred.write(str(dev_ent1s[i]))
                  fpred.write("\tArg2:T")
                  fpred.write(str(dev_ent2s[i]))
                  fpred.write("\n")
              elif dev_preds[i] == 4:
                  fpred.write(str(dev_pmids[i]))
                  fpred.write("\tCPR:6\tArg1:T")
                  fpred.write(str(dev_ent1s[i]))
                  fpred.write("\tArg2:T")
                  fpred.write(str(dev_ent2s[i]))
                  fpred.write("\n")
              elif dev_preds[i] == 5:
                  fpred.write(str(dev_pmids[i]))
                  fpred.write("\tCPR:9\tArg1:T")
                  fpred.write(str(dev_ent1s[i]))
                  fpred.write("\tArg2:T")
                  fpred.write(str(dev_ent2s[i]))
                  fpred.write("\n")
          fpred.close()

          flogits = open("logits", "w")
          for i in range(len(dev_logits)):
            flogits.write(str(dev_logits[i]))
            flogits.write("\n")
          flogits.close()

          fpid = open("pidAndDrugs", "w")
          for i in range(len(dev_pmids)):
            fpid.write(str(dev_pmids[i]) + "\t" + str(dev_ent1s[i]) + "\t" + str(dev_ent2s[i]))
            fpid.write("\n")
          fpid.close()

          tf.contrib.summary.scalar("dev/loss", dev_loss)
          tf.contrib.summary.scalar("dev/accuracy", dev_frac_correct)
        elif iterations % config.log_every == 0:
          mean_loss_val = mean_loss.result()
          accuracy_val = accuracy.result()
          print(log_template.format(
              time.time() - start,
              epoch, iterations, 1 + batch_idx, train_len,
              100.0 * (1 + batch_idx) / train_len,
              mean_loss_val, " " * 8, accuracy_val * 100.0, " " * 12))
          tf.contrib.summary.scalar("train/loss", mean_loss_val)
          tf.contrib.summary.scalar("train/accuracy", accuracy_val)
          # Reset metrics.
          mean_loss = tfe.metrics.Mean()
          accuracy = tfe.metrics.Accuracy()

        batch_idx += 1


def main(_):
  config = FLAGS
  # Load embedding vectors.
  vocab = data_chemprot.load_vocabulary("chemprot-data")

  ft1 = open('shorten_bc6/pubpmc.pickle', 'rb')
  embedding_for_given_index1 = pickle.load(ft1)
  ft1.close()
  word2index, embed = embedding_for_given_index1

  print("Loading train, dev and test data...")

  if not FLAGS.test_bool:
      train_data = data_chemprot.ChemprotData(
          os.path.join("chemprot-data", "train_whole_SNLIformat"),#training_SNLIformat
          word2index, sentence_len_limit=FLAGS.sentence_len_limit)
      dev_data = data_chemprot.ChemprotData(
          os.path.join("chemprot-data", "test_SNLIformat"),#develop_SNLIformat
          word2index, sentence_len_limit=FLAGS.sentence_len_limit)
      train_spinn(embed, train_data, dev_data, config)
  else:
      test_data = data_chemprot.ChemprotData(
          os.path.join("chemprot-data", "test_SNLIformat"),
      word2index, sentence_len_limit=FLAGS.sentence_len_limit)
      test_spinn(embed, test_data, config)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description=
      "TensorFlow eager implementation of the SPINN ChemprotClassifier.")
  parser.add_argument("--data_root", type=str, default="spinn-data",
                      help="Root directory in which the training data and "
                      "embedding matrix are found. See README.md for how to "
                      "generate such a directory.")
  parser.add_argument("--sentence_len_limit", type=int, default=250,
                      help="Maximum allowed sentence length (# of words). "
                      "The default of -1 means unlimited.")
  parser.add_argument("--logdir", type=str, default="tmpLog",
                      help="Directory in which summaries will be written for "
                      "TensorBoard.")
  parser.add_argument("--epochs", type=int, default=50,
                      help="Number of epochs to train.")
  parser.add_argument("--batch_size", type=int, default=256,
                      help="Batch size to use during training.")
  parser.add_argument("--d_proj", type=int, default=400,
                      help="Dimensions to project the word embedding vectors "
                      "to.")#it should be the double of the size of the word embedding vector
  parser.add_argument("--d_hidden", type=int, default=200,
                      help="Size of the hidden layer of the Tracker.")
  parser.add_argument("--d_out", type=int, default=6,
                      help="Output dimensions of the ChemprotClassifier.")
  parser.add_argument("--d_mlp", type=int, default=1024,
                      help="Size of each layer of the multi-layer perceptron "
                      "of the ChemprotClassifier.")
  parser.add_argument("--n_mlp_layers", type=int, default=2,
                      help="Number of layers in the multi-layer perceptron "
                      "of the ChemprotClassifier.")
  parser.add_argument("--d_tracker", type=int, default=64,#None better
                      help="Size of the tracker LSTM.")
  parser.add_argument("--log_every", type=int, default=50,
                      help="Print log and write TensorBoard summary every _ "
                      "training batches.")
  parser.add_argument("--lr", type=float, default=2e-3,
                      help="Initial learning rate.")
  parser.add_argument("--lr_decay_by", type=float, default=0.75,
                      help="The ratio to multiply the learning rate by every "
                      "time the learning rate is decayed.")
  parser.add_argument("--lr_decay_every", type=float, default=1,
                      help="Decay the learning rate every _ epoch(s).")
  parser.add_argument("--dev_every", type=int, default=1000,
                      help="Run evaluation on the dev split every _ training "
                      "batches.")
  parser.add_argument("--save_every", type=int, default=1000,
                      help="Save checkpoint every _ training batches.")
  parser.add_argument("--embed_dropout", type=float, default=0.08,#None better
                      help="Word embedding dropout rate.")
  parser.add_argument("--mlp_dropout", type=float, default=0.5,
                      help="ChemprotClassifier multi-layer perceptron dropout "
                      "rate.")
  parser.add_argument("--no-projection", action="store_false",
                      dest="projection",
                      help="Whether word embedding vectors are projected to "
                      "another set of vectors (see d_proj).")
  parser.add_argument("--predict_transitions", action="store_true",
                      dest="predict",
                      help="Whether the Tracker will perform prediction.")
  parser.add_argument("--force_cpu", action="store_true", dest="force_cpu",
                      help="Force use CPU-only regardless of whether a GPU is "
                      "available.")
  parser.add_argument("--test_bool", action="store_true", dest="test_bool",
                      help="For test")

  FLAGS, unparsed = parser.parse_known_args()

  tfe.run(main=main, argv=["--data_root chemprot-data --logdir tmpLog"])

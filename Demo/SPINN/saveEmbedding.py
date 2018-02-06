
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import tensorflow.contrib.eager as tfe
import my_data
from sklearn import metrics
import pickle

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description=
      "TensorFlow eager implementation of the SPINN chemprot classifier.")
  parser.add_argument("--data_root", type=str, default="spinn-data",
                      help="Root directory in which the training data and "
                      "embedding matrix are found. See README.md for how to "
                      "generate such a directory.")
  parser.add_argument("--sentence_len_limit", type=int, default=-1,
                      help="Maximum allowed sentence length (# of words). "
                      "The default of -1 means unlimited.")
  parser.add_argument("--logdir", type=str, default="tmpLog",
                      help="Directory in which summaries will be written for "
                      "TensorBoard.")
  parser.add_argument("--epochs", type=int, default=50,
                      help="Number of epochs to train.")
  parser.add_argument("--batch_size", type=int, default=256,
                      help="Batch size to use during training.")
  parser.add_argument("--d_proj", type=int, default=600,
                      help="Dimensions to project the word embedding vectors "
                      "to.")
  parser.add_argument("--d_hidden", type=int, default=300,
                      help="Size of the hidden layer of the Tracker.")
  parser.add_argument("--d_out", type=int, default=6,
                      help="Output dimensions of the ChemprotClassifier.")
  parser.add_argument("--d_mlp", type=int, default=1024,
                      help="Size of each layer of the multi-layer perceptron "
                      "of the ChemprotClassifier.")
  parser.add_argument("--n_mlp_layers", type=int, default=2,
                      help="Number of layers in the multi-layer perceptron "
                      "of the ChemprotClassifier.")
  parser.add_argument("--d_tracker", type=int, default=64,
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
  parser.add_argument("--embed_dropout", type=float, default=0.08,
                      help="Word embedding dropout rate.")
  parser.add_argument("--mlp_dropout", type=float, default=0.07,
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
  FLAGS, unparsed = parser.parse_known_args()

  config = FLAGS
  # Load embedding vectors.
  vocab = my_data.load_vocabulary("chemprot-data")
  word2index, embed, index2word = my_data.load_bioWE_vectors(vocab)

  embedding_for_given_index = (word2index, embed)
  ft = open('shorten_bc6/pubpmc.pickle', 'wb')
  pickle.dump(embedding_for_given_index, ft)
  ft.close()

  print(index2word[10], len(embed[10]))

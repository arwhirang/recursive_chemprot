# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities of ChemprotData and PubPmc word vectors for SPINN model.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import math
import os
import random

import copy
from gensim import models
import numpy as np

UNK_CODE = 0   # Code for unknown word tokens.
PAD_CODE = 1   # Code for padding tokens.

SHIFT_CODE = 3
REDUCE_CODE = 2

WORD_VECTOR_LEN = 200  # Embedding dimensions.

LEFT_PAREN = "("
RIGHT_PAREN = ")"
PARENTHESES = (LEFT_PAREN, RIGHT_PAREN)

def get_shift_reduce(items):
  """Obtain shift-reduce vector from a list of items from the SNLI-format data.

  Args:
    items: Data items as a list of str, e.g.,
       ["(", "Man", "(", "(", "(", "(", "(", "wearing", "pass", ")", ...

  Returns:
    A list of shift-reduce transitions, encoded as `SHIFT_CODE` for shift and
      `REDUCE_CODE` for reduce. See code above for the values of `SHIFT_CODE`
      and `REDUCE_CODE`.
  """
  trans = []
  for item in items:
    if item == LEFT_PAREN:
      continue
    elif item == RIGHT_PAREN:
      trans.append(REDUCE_CODE)
    else:
      trans.append(SHIFT_CODE)
  return trans


def pad_and_reverse_word_ids(sentences):
  """Pad a list of sentences to the common maximum length + 1.

  Args:
    sentences: A list of sentences as a list of list of integers. Each integer
      is a word ID. Each list of integer corresponds to one sentence.

  Returns:
    A numpy.ndarray of shape (num_sentences, max_length + 1), wherein max_length
      is the maximum sentence length (in # of words). Each sentence is reversed
      and then padded with an extra one at head, as required by the model.
  """
  max_len = max(len(sent) for sent in sentences)
  for sent in sentences:
    if len(sent) < max_len:
      sent.extend([PAD_CODE] * (max_len - len(sent)))
  # Reverse in time order and pad an extra one.
  sentences = np.fliplr(np.array(sentences, dtype=np.int64))
  sentences = np.concatenate(
      [np.ones([sentences.shape[0], 1], dtype=np.int64), sentences], axis=1)
  return sentences


def pad_transitions(sentences_transitions):
  """Pad a list of shift-reduce transitions to the maximum length."""
  max_len = max(len(transitions) for transitions in sentences_transitions)
  for transitions in sentences_transitions:
    if len(transitions) < max_len:
      transitions.extend([PAD_CODE] * (max_len - len(transitions)))
  return np.array(sentences_transitions, dtype=np.int64)


def load_vocabulary(data_root):
  """Load vocabulary from chemprot data files.

  Args:
    data_root: Root directory of the data.

  Returns:
    Vocabulary as a set of strings.

  Raises:
    ValueError: If chemprot data files cannot be found.
  """
  glob_pattern = os.path.join(data_root, "*")
  file_names = glob.glob(glob_pattern)
  if not file_names:
    raise ValueError(
        "Cannot find data files at %s. "
        "Please download and extract data first." % glob_pattern)

  print("Loading vocabulary...")
  vocab = set()
  wordAndCount = dict()
  for file_name in file_names:
    with open(file_name, "rt") as f:
      for i, line in enumerate(f):
        if i == 0:
          continue
        items = line.split("\t")
        justwords = items[2].split(", ")
        for word in justwords:
          if word in wordAndCount:
            wordAndCount[word] = wordAndCount[word] + 1
          else:
            wordAndCount[word] = 1

  for keyword in wordAndCount:
    if wordAndCount[keyword] > 3:
      vocab.add(keyword)
  wordAndCount.clear()
        #vocab.update(justwords)
  return vocab

def load_bioWE_vectors(vocab):
  """Load bioWE word vectors for words present in the vocabulary.

  Args:
    data_root: Data root directory for bioWE file.

  Returns:
    1. word2index: A dict from lower-case word to row index in the embedding
       matrix, i.e, `embed` below.
    2. embedding_for_given_index: The embedding matrix as a float32 numpy array. Its shape is
       [vocabulary_size, WORD_VECTOR_LEN]. vocabulary_size is len(vocab).
       WORD_VECTOR_LEN is the embedding dimension (200).

  """
  bioEmbed = models.KeyedVectors.load_word2vec_format('bioWordEmbedding/PubMed-and-PMC-w2v.bin', binary=True)

  print("Loading word vectors...")
  print("initializing with random vectors...")
  index2word = ["" for x in range(len(vocab)+2)]
  word2index = dict()
  word2index["<unk>"] = UNK_CODE
  word2index["<pad>"] = PAD_CODE
  wordi = 2#except unk and pad
  for strWord in vocab:
    index2word[wordi] = strWord
    word2index[strWord] = wordi
    wordi = wordi + 1
  init_range = math.sqrt(6.0 / (1 + 200))
  embedding_for_given_index = np.random.uniform(-init_range, init_range,
                                                (len(vocab)+2, 200)).astype(np.float32)
  count = 0
  for strWord in index2word:
    if strWord in bioEmbed:
      embedding_for_given_index[count] = np.float32(copy.deepcopy(bioEmbed[strWord]))
    count = count + 1

  return word2index, embedding_for_given_index, index2word

def calculate_bins(length2count, min_bin_size):
  """Cacluate bin boundaries given a histogram of lengths and mininum bin size.

  Args:
    length2count: A `dict` mapping length to sentence count.
    min_bin_size: Minimum bin size in terms of total number of sentence pairs
      in the bin.

  Returns:
    A `list` representing the right bin boundaries, starting from the inclusive
    right boundary of the first bin. For example, if the output is
      [10, 20, 35],
    it means there are three bins: [1, 10], [11, 20] and [21, 35].
  """
  bounds = []
  lengths = sorted(length2count.keys())
  cum_count = 0
  for length in lengths:
    cum_count += length2count[length]
    if cum_count >= min_bin_size:
      bounds.append(length)
      cum_count = 0
  if bounds[-1] != lengths[-1]:
    bounds.append(lengths[-1])
  return bounds


class ChemprotData(object):
  """A split of Chemprot data."""

  def __init__(self, data_file, word2index, sentence_len_limit=-1):
    """ChemprotData constructor.

    Args:
      data_file: Full path to the data file, e.g.,
        "/chemprot-data/developPosit_chem"
      word2index: A dict from lower-case word to row index in the embedding
        matrix (see `load_word_vectors()` for details).
      sentence_len_limit: Maximum allowed sentence length (# of words).
        A value of <= 0 means unlimited. Sentences longer than this limit
        are currently discarded, not truncated.
    """

    self._labels = []
    self._wordids = []
    self._transitions = []
    self._pmid = []
    self._ent1s = []
    self._ent2s = []

    with open(data_file, "rt") as f:
      for i, line in enumerate(f):
        items = line.split("\t")

        parsed = items[1].split(" ")#parsed sentences
        justwords = items[2].split(", ")

        if (sentence_len_limit > 0 and
            (len(justwords) > sentence_len_limit)):
          continue

        word_ids_list = [
            word2index.get(word, UNK_CODE) for word in justwords]

        self._wordids.append(word_ids_list)
        self._transitions.append(get_shift_reduce(parsed))
        assert (len(self._transitions[-1]) ==
                2 * len(justwords) - 1)

        self._labels.append(int(items[0]))
        self._pmid.append(int(items[3]))
        self._ent1s.append(int(items[4][1:]))
        self._ent2s.append(int(items[5][1:-1]))

    assert len(self._labels) == len(self._wordids)
    assert len(self._labels) == len(self._transitions)
  
  def num_batches(self, batch_size):
    """Calculate number of batches given batch size."""
    return int(math.ceil(len(self._labels) / batch_size))

  def get_generator(self, batch_size):
    """Obtain a generator for batched data.

    All examples of this data object are randomly shuffled, sorted
    according to the maximum sentence length of the sentences in the pair, and batched.

    Args:
      batch_size: Desired batch size.

    Returns:
      A generator for data batches. The generator yields a 3-tuple:
        label: An array of the shape (batch_size,).
        sentence_ids: An array of the shape (max_premise_len, batch_size), wherein
          max_sen_len is the maximum length of the (padded) sentence in the batch.
        transitions: An array of the shape (2 * max_sen_len -3, batch_size).
      All the elements of the 3-tuple have dtype `int64`.
    """
    # Randomly shuffle examples.
    zipped = list(zip(
        self._labels, self._wordids, self._transitions, self._pmid, self._ent1s, self._ent2s))
    random.shuffle(zipped)
    # Then sort the examples by sentence lengths.
    # During training, the batches are expected to be shuffled.
    # So it is okay to leave them sorted by max length here.
    (labels, wordids, transitions, pmids, ent1s, ent2s) = zip(
         *sorted(zipped, key=lambda x: len(x[1])))

    def _generator():
      begin = 0
      while begin < len(labels):
        # The sorting above and the batching here makes sure that sentences of
        # similar max lengths are batched together, minimizing the inefficiency
        # due to uneven max lengths. The sentences are batched differently in
        # each call to get_generator() due to the shuffling before sotring
        # above. The pad_and_reverse_word_ids() and pad_transitions() functions
        # take care of any remaning unevenness of the max sentence lengths.
        end = min(begin + batch_size, len(labels))
        # Transpose, because the SPINN model requires time-major, instead of
        # batch-major.
        yield (labels[begin:end],
               pad_and_reverse_word_ids(wordids[begin:end]).T,
               pad_transitions(transitions[begin:end]).T,
               pmids[begin:end],
               ent1s[begin:end],
               ent2s[begin: end])
        begin = end
    return _generator

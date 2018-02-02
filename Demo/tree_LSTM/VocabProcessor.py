# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#
# I modified this code from tensorflow.contrib learn.preprocessing.VocabularyProcessor
# 
# ==============================================================================

import re
import numpy as np
import six
import nltk
import collections
import six
from tensorflow.python.platform import gfile
try:
  # pylint: disable=g-import-not-at-top
  import cPickle as pickle
except ImportError:
  # pylint: disable=g-import-not-at-top
  import pickle

class CategoricalVocabulary(object):
  """Categorical variables vocabulary class.
  Accumulates and provides mapping from classes to indexes.
  Can be easily used for words.
  """

  def __init__(self, unknown_token="<UNK>", support_reverse=True):
    self._unknown_token = unknown_token
    self._mapping = {unknown_token: 0}
    self._support_reverse = support_reverse
    if support_reverse:
      self._reverse_mapping = [unknown_token]
    self._freq = collections.defaultdict(int)
    self._freeze = False
  
  def __len__(self):
    """Returns total count of mappings. Including unknown token."""
    return len(self._mapping)

  def freeze(self, freeze=True):
    """Freezes the vocabulary, after which new words return unknown token id.
    Args:
      freeze: True to freeze, False to unfreeze.
    """
    self._freeze = freeze
  
  def get(self, category):
    """Returns word's id in the vocabulary.
    If category is new, creates a new id for it.
    Args:
      category: string or integer to lookup in vocabulary.
    Returns:
      interger, id in the vocabulary.
    """
    if category not in self._mapping:
        if self._freeze:
            return 0
        self._mapping[category] = len(self._mapping)
        if self._support_reverse:
            self._reverse_mapping.append(category)
    return self._mapping[category]

  def add(self, category, count=1):
    """Adds count of the category to the frequency table.
    Args:
      category: string or integer, category to add frequency to.
      count: optional integer, how many to add.
    """
    category_id = self.get(category)
    if category_id <= 0:
        return
    self._freq[category] += count
  
  def trim(self, min_frequency, max_frequency=-1):
    """Trims vocabulary for minimum frequency.
    Remaps ids from 1..n in sort frequency order.
    where n - number of elements left.
    Args:
      min_frequency: minimum frequency to keep.
      max_frequency: optional, maximum frequency to keep.
        Useful to remove very frequent categories (like stop words).
    """
    # Sort by alphabet then reversed frequency.
    self._freq = sorted(
        sorted(
            six.iteritems(self._freq),
            key=lambda x: (isinstance(x[0], str), x[0])),
        key=lambda x: x[1],
        reverse=True)
    self._mapping = {self._unknown_token: 0}
    if self._support_reverse:
      self._reverse_mapping = [self._unknown_token]
    idx = 1
    for category, count in self._freq:
        if max_frequency > 0 and count >= max_frequency:
            continue
        if count <= min_frequency:
            print(category, count, min_frequency)
            continue
        self._mapping[category] = idx
        idx += 1
        if self._support_reverse:
            self._reverse_mapping.append(category)
    self._freq = dict(self._freq[:idx - 1])
  
  def reverse(self, class_id):
    """Given class id reverse to original class name.
    Args:
      class_id: Id of the class.
    Returns:
      Class name.
    Raises:
      ValueError: if this vocabulary wasn't initalized with support_reverse.
    """
    if not self._support_reverse:
        raise ValueError("This vocabulary wasn't initalized with "
                       "support_reverse to support reverse() function.")
    return self._reverse_mapping[class_id]


class VocabProcessor(object):
  """Maps documents to sequences of word ids."""

  def __init__(self,
               max_document_length,
               min_frequency=0,
               vocabulary=None,
               tokenizer_fn=None):
    """Initializes a VocabularyProcessor instance.
    Args:
      max_document_length: Maximum length of documents.
        if documents are longer, they will be trimmed, if shorter - padded.
      min_frequency: Minimum frequency of words in the vocabulary.
      vocabulary: CategoricalVocabulary object.
    Attributes:
      vocabulary_: CategoricalVocabulary object.
    """
    self.max_document_length = max_document_length
    self.min_frequency = min_frequency
    if vocabulary:
      self.vocabulary_ = vocabulary
    else:
      self.vocabulary_ = CategoricalVocabulary()
    if tokenizer_fn == "split":
      self._tokenizer = "split"
    elif tokenizer_fn == "splitComma":
      self._tokenizer = "splitComma"
    elif tokenizer_fn == "nltk":
      self._tokenizer = "nltk"
  
  def fit(self, raw_documents):
    """Learn a vocabulary dictionary of all tokens in the raw documents.
    Args:
      raw_documents: An iterable which yield either str or unicode.
    Returns:
      self
    """
    if self._tokenizer == "nltk":
        for sentence in raw_documents:
            for tokens in nltk.word_tokenize(sentence):
                self.vocabulary_.add(tokens)
    elif self._tokenizer == "split":
        for sentence in raw_documents:
            for tokens in sentence.split(" "):
                self.vocabulary_.add(tokens)
    elif self._tokenizer == "splitComma":
        for sentence in raw_documents:
            for tokens in sentence.split(", "):
                self.vocabulary_.add(tokens)
    if self.min_frequency > 0:
        self.vocabulary_.trim(self.min_frequency)
    self.vocabulary_.freeze()
    return self

  def fit_transform(self, raw_documents, unused_y=None):
    """Learn the vocabulary dictionary and return indexies of words.
    Args:
      raw_documents: An iterable which yield either str or unicode.
      unused_y: to match fit_transform signature of estimators.
    Returns:
      x: iterable, [n_samples, max_document_length]. Word-id matrix.
    """
    self.fit(raw_documents)
    return self.transform(raw_documents)

  def transform(self, raw_documents):
    """Transform documents to word-id matrix.
    Convert words to ids with vocabulary fitted with fit or the one
    provided in the constructor.
    Args:
      raw_documents: An iterable which yield either str or unicode.
    Yields:
      x: iterable, [n_samples, max_document_length]. Word-id matrix.
    """
    if self._tokenizer == "nltk":
        for sentence in raw_documents:
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(nltk.word_tokenize(sentence)):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield word_ids
    elif self._tokenizer == "split":
        for sentence in raw_documents:
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(sentence.split(" ")):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield word_ids
    elif self._tokenizer == "splitComma":
        for sentence in raw_documents:
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(sentence.split(", ")):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield word_ids
  
  def wordToNumber(self, word):
    if word == '':
        return 0
    else:
        return self.vocabulary_.get(word)
  
  def reverse(self, documents):
    """Reverses output of vocabulary mapping to words.
    Args:
      documents: iterable, list of class ids.
    Yields:
      Iterator over mapped in words documents.
    """
    for item in documents:
        output = []
        for class_id in item:
            output.append(self.vocabulary_.reverse(class_id))
        yield ' '.join(output)
  
  def save(self, filename):
    """Saves vocabulary processor into given file.
    Args:
      filename: Path to output file.
    """
    with gfile.Open(filename, 'wb') as f:
      f.write(pickle.dumps(self))

  @classmethod
  def restore(cls, filename):
    """Restores vocabulary processor from given file.
    Args:
      filename: Path to file to load from.
    Returns:
      VocabularyProcessor object.
    """
    with gfile.Open(filename, 'rb') as f:
        return pickle.loads(f.read())

import tensorflow as tf
import numpy as np
from gensim import models
from VocabProcessor import VocabProcessor
import re
import nltk
import math
import pickle

# Model Hyperparameters
# ==================================================
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
# Training parameters
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\"", "", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    #     string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"/", " / ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"  ", " ", string)
    return string.lower()

def separateFeatures(string):
    for line in string:
        # line = clean_str(line)
        line = line.replace("ddi-drug", "ddrug")
        pid = line.split("\t")[0]
        sen = line.split("\t")[1]
        bc6Check = line.split("\t")[2]
        bc6Type = line.split("\t")[3]
        drug1 = line.split("\t")[4]
        drug1Name = line.split("\t")[5]
        gene2 = line.split("\t")[6]
        gene2Name = line.split("\t")[7]
        binaryParsedTree = line.split("\t")[8].strip()
        parsedWholeSen = line.split("\t")[9]
        yield binaryParsedTree, parsedWholeSen, drug1, gene2

def load_data_and_labels(string):
    """
    Loads data from files, splits the data into words and relative
    drug position and 6 labels.
    all charancters are lowercased.
    Returns split sentences
    """
    # Load data from files
    samples = list(open(string, "r").readlines())
    return list(separateFeatures(samples))

# Data Preparatopn
# ==================================================
TrainFeatures = load_data_and_labels("dataPreprocessor/trainingPosit_chem")
DevFeatures = load_data_and_labels("dataPreprocessor/developPosit_chem")
TestFeatures = load_data_and_labels("dataPreprocessor/testPosit_chem")

allSens1 = [Tf[1] for Tf in TrainFeatures]+ [Tf[1] for Tf in DevFeatures]+ [Tf[1] for Tf in TestFeatures]
splitted1 = [sentence.split(", ") for sentence in allSens1]
max_document_length1 = max(len(s) for s in splitted1)
vocab_proc1 = VocabProcessor(max_document_length1, tokenizer_fn="splitComma", min_frequency=3)
sens_train_whole = np.array(list(vocab_proc1.fit_transform([Tf[1] for Tf in TrainFeatures])))
sens_dev_whole = np.array(list(vocab_proc1.fit_transform([Tf[1] for Tf in DevFeatures])))
sens_test_whole = np.array(list(vocab_proc1.fit_transform([Tf[1] for Tf in TestFeatures])))


# #load bioWordEmbedding
# # ==================================================
word_for_given_index1 = ["" for x in range(len(vocab_proc1.vocabulary_))]
for strWord in vocab_proc1.vocabulary_._mapping:
    word_for_given_index1[vocab_proc1.vocabulary_.get(strWord)] = strWord

bioEmbed = models.KeyedVectors.load_word2vec_format('bioWordEmbedding/PubMed-and-PMC-w2v.bin', binary=True)
count = 0
init_range = math.sqrt(6.0 / (1 + 200))
embedding_for_given_index = np.random.uniform(-init_range, init_range,
                                               (len(vocab_proc1.vocabulary_), FLAGS.embedding_dim)).astype(np.float32)
for strWord in word_for_given_index1:
    if strWord in bioEmbed:
        embedding_for_given_index[count] = np.float32(bioEmbed[strWord])
    count = count + 1

ft = open('shorten_bc6/pubpmc_test.pickle', 'wb')
pickle.dump(embedding_for_given_index, ft)
ft.close()

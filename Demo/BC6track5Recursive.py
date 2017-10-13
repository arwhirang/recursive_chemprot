# This code is based on the treelstm code
# https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/sentiment.ipynb
import os
from VocabProcessor import VocabProcessor
import re
from sklearn import metrics
from nltk.tokenize import sexpr
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
import tensorflow_fold as td
import pickle
import sys
sys.setrecursionlimit(2000)

# Model Hyperparameters
# ==================================================
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
# Training parameters
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


# Tokenization/string cleaning for all datasets except for SST.
# Original code was taken & modified from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
# ==================================================
def clean_str(string):
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"  ", " ", string)
    return string.lower()


def separateFeatures(string):
    for line in string:
        line = clean_str(line)
        pid = line.split("\t")[0]
        sen = line.split("\t")[1]
        bc6Check = line.split("\t")[2]
        drug1 = line.split("\t")[4]
        drug2 = line.split("\t")[6]
        binaryParsedTree = line.split("\t")[8].strip()
        parsedWholeSen = line.split("\t")[9]

        yield binaryParsedTree, parsedWholeSen


def load_data_and_labels(string):
    """
    Loads BioCreative Challenge6 track5 data from files.
    target entities are anonymized.
    """
    # Load data from files
    samples = list(open(string, "r").readlines())
    return list(separateFeatures(samples))

data_dir = "./save"
print('saving files to %s' % data_dir)
TrainFeatures = load_data_and_labels("data/trainingPosit_new")
DevFeatures = load_data_and_labels("data/developPosit_new")
TestFeatures = load_data_and_labels("data/testPosit_new")

train_trees = [Tf[0] for Tf in TrainFeatures] + [Tf[0] for Tf in DevFeatures]
#train_trees = [Tf[0] for Tf in TrainFeatures]
dev_trees = [Tf[0] for Tf in DevFeatures]
#test_trees = dev_trees
test_trees = [Tf[0] for Tf in TestFeatures]

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(TrainFeatures)))
train_trees = np.array(train_trees)[shuffle_indices]

print("length of train : ",len(train_trees))
print("length of dev : ",len(dev_trees))

allSens1 = [Tf[1] for Tf in TrainFeatures]+ [Tf[1] for Tf in DevFeatures]+ [Tf[1] for Tf in TestFeatures]
#allSens1 = [Tf[1] for Tf in TrainFeatures]+ [Tf[1] for Tf in DevFeatures]
splitted1 = [sentence.split(", ") for sentence in allSens1]
max_document_length1 = max(len(s) for s in splitted1)
vocab_proc1 = VocabProcessor(max_document_length1, tokenizer_fn="splitComma")
sens_train_whole = np.array(list(vocab_proc1.fit_transform([Tf[1] for Tf in TrainFeatures])))
sens_dev_whole = np.array(list(vocab_proc1.fit_transform([Tf[1] for Tf in DevFeatures])))
sens_test_whole = np.array(list(vocab_proc1.fit_transform([Tf[1] for Tf in TestFeatures])))
vocab_proc1.vocabulary_.freeze()

ft1 = open('shorten_bc6/pubpmc_real.pickle', 'rb')
embedding_for_given_index1 = pickle.load(ft1)
ft1.close()

print(len(embedding_for_given_index1))

weight_matrix = embedding_for_given_index1
word_idx = vocab_proc1.vocabulary_

# reading parameters end
# =====================
class BinaryTreeLSTMCell(tf.contrib.rnn.RNNCell):
    """
    LSTM with two state inputs.

    This model is based on the model of 'Improved Semantic
    Representations From Tree-Structured Long Short-Term Memory
    Networks' <http://arxiv.org/pdf/1503.00075.pdf>, with recurrent
    dropout as described in 'Recurrent Dropout without Memory Loss'
    <http://arxiv.org/pdf/1603.05118.pdf>.

    Original code was retrieved from the tensorflow fold library.
    """
    def __init__(self, num_units, keep_prob=1.0):
        """Initialize the cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          keep_prob: Keep probability for recurrent dropout.
        """
        super(BinaryTreeLSTMCell, self).__init__()
        self._keep_prob = keep_prob
        self._num_units = num_units
        self.state_size = (num_units, num_units)
        self.output_size = num_units * 1

    def state_size(self):
        self.state_size = (self._num_units, self._num_units)

    def output_size(self):
        self.output_size = (self._num_units * 1)

    def __call__(self, inputs, state, contextVec, ent1Vec, ent2Vec, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            inputs = tf.nn.dropout(inputs, self._keep_prob)
            lhs, rhs = state
            c_0, h_0 = lhs
            c_1, h_1 = rhs
            # first (binary tree) lstm
            concat0 = tf.contrib.layers.fully_connected(
                tf.concat([contextVec, ent1Vec, ent2Vec, inputs, h_0, h_1], 1), 5 * self._num_units, trainable=True)
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i_0, j_0, f_00, f_01, o_0 = tf.split(value=concat0, num_or_size_splits=5, axis=1)
            j_0 = tf.tanh(j_0)
            if not isinstance(self._keep_prob, float) or self._keep_prob < 1:
                j_0 = tf.nn.dropout(j_0, self._keep_prob)
            new_c0 = (c_0 * tf.sigmoid(f_00 + 1.0) +
                      c_1 * tf.sigmoid(f_01 + 1.0) +
                      tf.sigmoid(i_0) * j_0)
            new_h0 = tf.tanh(new_c0) * tf.sigmoid(o_0)

            resultH = tf.concat([new_h0], 1)
            resultH = tf.nn.dropout(resultH, self._keep_prob)

            return resultH, [new_c0, new_h0]


# dropout keep probability, with a default of 1 (for eval).
keep_prob_ph = tf.placeholder_with_default(1.0, [])

lstm_num_units = 256  # Tai et al. used 150
tree_lstm = td.ScopedLayer(
    BinaryTreeLSTMCell(lstm_num_units, keep_prob=keep_prob_ph),
    name_or_scope='tree_lstm')
NUM_CLASSES = 6  # number of distinct labels
output_layer = td.FC(NUM_CLASSES, activation=None, name='output_layer')

word_embedding = td.Embedding(
    *weight_matrix.shape, initializer=weight_matrix, name='word_embedding', trainable=False)

# declare recursive model
embed_subtree = td.ForwardDeclaration(name='embed_subtree')

def makeContextMat(input1):
    input1 = int(input1)
    if input1 == 0:
        #     if input1 < 2:
        return [1 for i in range(10)]
    else:
        return [0 for i in range(10)]

def makeDepthMat(input2):
    input1 = int(input2)
    return [1 if i < input1 else 0 for i in range(20)]

def makeEntPositMat(input2):
    position_embed = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                      [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                      [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    input1 = int(input2)
    return position_embed[input1]

def logits_and_state():
    """Creates a block that goes from tokens to (logits, state) tuples."""
    unknown_idx = len(word_idx)

    lookup_word = lambda word: word_idx.get(word)  # unknown_idx is the default return value
    word2vec = (td.GetItem(0) >> td.GetItem(0) >> td.InputTransform(lookup_word) >>
                td.Scalar('int32') >> word_embedding)  # <td.Pipe>: None -> TensorType((200,), 'float32')
    context2vec1 = td.GetItem(1) >> td.InputTransform(makeContextMat) >> td.Vector(10)
    context2vec2 = td.GetItem(1) >> td.InputTransform(makeContextMat) >> td.Vector(10)
    ent1posit1 = td.GetItem(2) >> td.InputTransform(makeEntPositMat) >> td.Vector(10)
    ent1posit2 = td.GetItem(2) >> td.InputTransform(makeEntPositMat) >> td.Vector(10)
    ent2posit1 = td.GetItem(3) >> td.InputTransform(makeEntPositMat) >> td.Vector(10)
    ent2posit2 = td.GetItem(3) >> td.InputTransform(makeEntPositMat) >> td.Vector(10)

    pair2vec = td.GetItem(0) >> (embed_subtree(), embed_subtree())
    # Trees are binary, so the tree layer takes two states as its input_state.
    zero_state = td.Zeros((tree_lstm.state_size,) * 2)
    # Input is a word vector.
    zero_inp = td.Zeros(word_embedding.output_type.shape[0])  # word_embedding.output_type.shape[0] == 200

    word_case = td.AllOf(word2vec, zero_state, context2vec1, ent1posit1, ent2posit1)
    pair_case = td.AllOf(zero_inp, pair2vec, context2vec2, ent1posit2, ent2posit2)
    # if leaf case, go to word case...
    tree2vec = td.OneOf(lambda pair: len(pair[0]), [(1, word_case), (2, pair_case)])
    # logits and lstm states
    return tree2vec >> tree_lstm >> (output_layer, td.Identity())


# Define a per-node loss function for training.
def tf_node_loss(logits, labels):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return losses

def tf_hits(logits, labels):
    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    return tf.cast(tf.equal(predictions, labels), tf.float32)

def tf_pred(logits):
    return tf.cast(tf.argmax(logits, 1), tf.int32)

def tf_logits(logits):
    return logits

def tf_label(labels):
    return labels

def add_metrics(is_root):
    c = td.Composition(
        name='predict(is_root=%s)' % (is_root))
    with c.scope():
        labels = c.input[0]
        logits = td.GetItem(0).reads(c.input[1])
        state = td.GetItem(1).reads(c.input[1])

        loss = td.Function(tf_node_loss)
        td.Metric('all_loss').reads(loss.reads(logits, labels))
        if is_root:
            td.Metric('root_loss').reads(loss)

        result_logits = td.Function(tf_logits)
        td.Metric('all_logits').reads(result_logits.reads(logits))
        if is_root:
            td.Metric('root_logits').reads(result_logits)
        # reserve pred and labels
        pred = td.Function(tf_pred)
        td.Metric('all_pred').reads(pred.reads(logits))
        if is_root:
            td.Metric('root_pred').reads(pred)
        answer = td.Function(tf_label)
        td.Metric('all_labels').reads(answer.reads(labels))
        if is_root:
            td.Metric('root_label').reads(answer)

        c.output.reads(state)
    return c


def tokenize(s):
    labelAndDepth, phrase = s[1:-1].split(None, 1)
    label, outerContext, ent1Posit, ent2Posit = labelAndDepth.split("/")
    # classification
    return label, (sexpr.sexpr_tokenize(phrase), outerContext, ent1Posit, ent2Posit)

def embed_tree(is_root):
    return td.InputTransform(tokenize) >> (td.Scalar('int32'), logits_and_state()) >> add_metrics(is_root)

model = embed_tree(is_root=True)
# Resolve the forward declaration for embedding subtrees (the non-root case) with a second call to embed_tree.
embed_subtree.resolve_to(embed_tree(is_root=False))
print('input type: %s' % model.input_type)
print('output type: %s' % model.output_type)
compiler = td.Compiler.create(model)  # batching
# build model end
# ==================

pred = compiler.metric_tensors['root_pred']
labels = compiler.metric_tensors['root_label']
result_logits = compiler.metric_tensors['root_logits']

LEARNING_RATE = 0.001
KEEP_PROB = 0.5
BATCH_SIZE = 256
EPOCHS = 10000
EMBEDDING_LEARNING_RATE_FACTOR = 0.01

train_feed_dict = {keep_prob_ph: KEEP_PROB}
loss = tf.reduce_sum(compiler.metric_tensors['root_loss'])
opt = tf.train.AdamOptimizer(LEARNING_RATE)

grads_and_vars = opt.compute_gradients(loss)
train_op = opt.apply_gradients(grads_and_vars)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

def train_step(batch):
    train_feed_dict[compiler.loom_input_tensor] = batch
    _, batch_loss, train_logits, train_labels = sess.run([train_op, loss, result_logits, labels], train_feed_dict)
    return batch_loss

def train_epoch(train_set):
    return sum(train_step(batch) for batch in td.group_by_batches(train_set, BATCH_SIZE))

train_set = compiler.build_loom_inputs(train_trees)
dev_feed_dict = compiler.build_loom_inputs(dev_trees)
test_feed_dict = compiler.build_feed_dict(test_trees)

def dev_eval(epoch, train_loss):
    dev_loss = tf.reduce_sum(compiler.metric_tensors['root_loss'])
    dev_loss_whole = 0.
    dev_pred_whole = []
    dev_labels_whole = []
    dev_set = compiler.build_loom_inputs(dev_trees)
    for batch in td.group_by_batches(dev_set, BATCH_SIZE):
        dev_feed_dict[compiler.loom_input_tensor] = batch
        dev_loss_batch, dev_pred_batch, dev_labels_batch = sess.run([dev_loss, pred, labels], dev_feed_dict)
        dev_loss_whole = dev_loss_whole + dev_loss_batch
        dev_pred_whole = dev_pred_whole + dev_pred_batch.tolist()
        dev_labels_whole = dev_labels_whole + dev_labels_batch.tolist()
        
    f1score = metrics.f1_score(dev_labels_whole, dev_pred_whole, average=None)
    print('epoch:%4d, train_loss: %.3e, dev_loss_avg: %.3e, dev_f1score:\n  [%s]'
      % (epoch, train_loss, dev_loss_whole, f1score))
      
def test_eval():
    test_loss = tf.reduce_sum(compiler.metric_tensors['root_loss'])
    _test_logits = compiler.metric_tensors['root_logits']
    test_loss_whole = 0.
    test_pred_whole = []
    test_labels_whole = []
    test_logits_whole = []
    # f1 = open("logTmp", "w")
    for batch in td.group_by_batches(test_set, BATCH_SIZE):
        test_feed_dict[compiler.loom_input_tensor] = batch
        test_loss_batch,test_pred_batch, test_labels_batch, test_logits_batch = sess.run([test_loss, pred, labels, _test_logits], test_feed_dict)
        test_loss_whole = test_loss_whole + test_loss_batch
        test_pred_whole = test_pred_whole + test_pred_batch.tolist()
        test_labels_whole = test_labels_whole + test_labels_batch.tolist()
        test_logits_whole = test_logits_whole + test_logits_batch.tolist()
    f1score = metrics.f1_score(test_labels_whole, test_pred_whole, average=None)
    print('test_loss_avg: %.3e, test_f1score:\n  [%s]'
      % (test_loss_whole, f1score))
    return test_labels_whole, test_pred_whole, test_logits_whole
    
#Run the main training loop, saving the model after each epoch if it has the best f1 score on the dev set.
best_f1score = 0.0
save_path = os.path.join(data_dir, 'BC6_track5_model')
for epoch, shuffled in enumerate(td.epochs(train_set, EPOCHS), 1):
    train_loss = train_epoch(shuffled)
    print("epoch %s finished, train_loss: %.3e" % (epoch, train_loss))
    # f1score = dev_eval(epoch, train_loss)
    if epoch % 100 == 0:
        checkpoint_path = saver.save(sess, save_path, global_step=epoch)
        print('model saved in file: %s' % checkpoint_path)
"""
#for rough testing
saver.restore(sess, "save/BC6_track5_model-100")
test_labels_whole, test_pred_whole, test_logits_whole = test_eval()
"""

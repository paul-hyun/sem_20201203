# -*- coding:utf-8 -*-
import itertools
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sentencepiece as spm
import tensorflow as tf
import tensorflow.keras.backend as K
import wget

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
print(tf.test.gpu_device_name())

#
# prepare dir
#
data_dir = './data'
if not os.path.exists(data_dir):
    data_dir = '../data'
print(os.listdir(data_dir))

konli_dir = os.path.join(data_dir, 'konli')
if not os.path.exists(konli_dir):
    os.makedirs(konli_dir)

train_txt = os.path.join(konli_dir, 'snli_1.0_train.ko.tsv')
dev_txt = os.path.join(konli_dir, 'xnli.dev.ko.tsv')
test_txt = os.path.join(konli_dir, 'xnli.test.ko.tsv')

#
# download file
#

wget.download('https://github.com/kakaobrain/KorNLUDatasets/raw/master/KorNLI/snli_1.0_train.ko.tsv', train_txt)
wget.download('https://raw.githubusercontent.com/kakaobrain/KorNLUDatasets/master/KorNLI/xnli.dev.ko.tsv', dev_txt)
wget.download('https://raw.githubusercontent.com/kakaobrain/KorNLUDatasets/master/KorNLI/xnli.test.ko.tsv', test_txt)
print(os.listdir(konli_dir))


def print_file(filename, count=10):
    """
    라인 수 만큼 파일내용 출력
    :param filename: file name
    :param count: print line count
    """
    with open(filename) as f:
        for i, line in enumerate(f):
            print(line.strip())
            if count < i:
                break


print_file(train_txt)
print_file(dev_txt)
print_file(test_txt)

#
# data read
# https://pandas.pydata.org/pandas-docs/stable/index.html
#

# head=0 첫벗째 줄이 head, quoting=3 큰따옴표 무시
train_data = pd.read_csv(train_txt, header=0, delimiter='\t', quoting=3)
print(f'전체 학습 raw 개수: {len(train_data)}')
train_data = train_data.dropna()
print(f'전체 학습 valid 개수: {len(train_data)}')
train_data = train_data.sample(1000)  # 빠른 확인을 위해 1000개만 사용
print(f'전체 학습 sample 개수: {len(train_data)}')
label_counts = train_data['gold_label'].value_counts()
print(f'전체 학습 label 개수: {label_counts}')

# head=0 첫벗째 줄이 head, quoting=3 큰따옴표 무시
dev_data = pd.read_csv(dev_txt, header=0, delimiter='\t', quoting=3)
print(f'전체 검증 raw 개수: {len(dev_data)}')
dev_data = dev_data.dropna()
print(f'전체 검증 valid 개수: {len(dev_data)}')
dev_data = dev_data.sample(500)  # 빠른 확인을 위해 500개만 사용
print(f'전체 검증 sample 개수: {len(dev_data)}')
label_counts = dev_data['gold_label'].value_counts()
print(f'전체 검증 label 개수: {label_counts}')

# head=0 첫벗째 줄이 head, quoting=3 큰따옴표 무시
test_data = pd.read_csv(test_txt, header=0, delimiter='\t', quoting=3)
print(f'전체 시험 raw 개수: {len(test_data)}')
test_data = test_data.dropna()
print(f'전체 시험 valid 개수: {len(test_data)}')
test_data = test_data.sample(450)  # 빠른 확인을 위해 450개만 사용
print(f'전체 시험 sample 개수: {len(test_data)}')
label_counts = test_data['gold_label'].value_counts()
print(f'전체 시험 label 개수: {label_counts}')

#
# vocabulary
#

# vocab load
vocab_file = os.path.join(data_dir, 'ko_32000.model')
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

#
# tokenize
#
train_sentence1s, train_sentence2s, train_gold_labels = [], [], []
for i, row in train_data.iterrows():
    sentence1 = vocab.encode_as_pieces(row['sentence1'])
    train_sentence1s.append(sentence1)
    sentence2 = vocab.encode_as_pieces(row['sentence2'])
    train_sentence2s.append(sentence2)
    train_gold_labels.append(row['gold_label'])

assert len(train_sentence1s) == len(train_sentence2s) == len(train_gold_labels)

print(train_sentence1s[:100])
print(train_sentence2s[:100])
print(train_gold_labels[:100])

dev_sentence1s, dev_sentence2s, dev_gold_labels = [], [], []
for i, row in dev_data.iterrows():
    sentence1 = vocab.encode_as_pieces(row['sentence1'])
    dev_sentence1s.append(sentence1)
    sentence2 = vocab.encode_as_pieces(row['sentence2'])
    dev_sentence2s.append(sentence2)
    dev_gold_labels.append(row['gold_label'])

assert len(dev_sentence1s) == len(dev_sentence2s) == len(dev_gold_labels)

print(dev_sentence1s[:100])
print(dev_sentence2s[:100])
print(dev_gold_labels[:100])

test_sentence1s, test_sentence2s, test_gold_labels = [], [], []
for i, row in test_data.iterrows():
    sentence1 = vocab.encode_as_pieces(row['sentence1'])
    test_sentence1s.append(sentence1)
    sentence2 = vocab.encode_as_pieces(row['sentence2'])
    test_sentence2s.append(sentence2)
    test_gold_labels.append(row['gold_label'])

assert len(test_sentence1s) == len(test_sentence2s) == len(test_gold_labels)

print(test_sentence1s[:100])
print(test_sentence2s[:100])
print(test_gold_labels[:100])

#
# token to id
#
label_to_id = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
print(label_to_id)
id_to_label = {value: key for key, value in label_to_id.items()}
print(id_to_label)

train_sentence1_ids = [[vocab.piece_to_id(p) for p in token] for token in train_sentence1s]
print(train_sentence1_ids[:100])
train_sentence2_ids = [[vocab.piece_to_id(p) for p in token] for token in train_sentence2s]
print(train_sentence2_ids[:100])
train_label_ids = [label_to_id[gold_label] for gold_label in train_gold_labels]
print(train_label_ids[:100])
assert len(train_sentence1_ids) == len(train_sentence2_ids) == len(train_label_ids)

dev_sentence1_ids = [[vocab.piece_to_id(p) for p in token] for token in dev_sentence1s]
print(dev_sentence1_ids[:100])
dev_sentence2_ids = [[vocab.piece_to_id(p) for p in token] for token in dev_sentence2s]
print(dev_sentence2_ids[:100])
dev_label_ids = [label_to_id[gold_label] for gold_label in dev_gold_labels]
print(dev_label_ids[:100])
assert len(dev_sentence1_ids) == len(dev_sentence2_ids) == len(dev_label_ids)

test_sentence1_ids = [[vocab.piece_to_id(p) for p in token] for token in test_sentence1s]
print(test_sentence1_ids[:100])
test_sentence2_ids = [[vocab.piece_to_id(p) for p in token] for token in test_sentence2s]
print(test_sentence2_ids[:100])
test_label_ids = [label_to_id[gold_label] for gold_label in test_gold_labels]
print(test_label_ids[:100])
assert len(test_sentence1_ids) == len(test_sentence2_ids) == len(test_label_ids)

#
# pad
#

# 길이가 달라서 matrix 생성 안됨
print(np.array(train_sentence1_ids)[:50])
print(np.array(train_sentence2_ids)[:50])

# train token 길이 확인
train_sentence_all = train_sentence1_ids + train_sentence2_ids
train_token_length = [len(token_id) for token_id in train_sentence_all]
print(train_token_length[:100])

# dev token 길이 확인
dev_sentence_all = dev_sentence1_ids + dev_sentence2_ids
dev_token_length = [len(token_id) for token_id in dev_sentence_all]
print(dev_token_length[:100])

# test token 길이 확인
test_sentence_all = test_sentence1_ids + test_sentence2_ids
test_token_length = [len(token_id) for token_id in test_sentence_all]
print(test_token_length[:100])

# 최대 길이 확인
train_max_length, dev_max_length, test_max_length = max(train_token_length), max(dev_token_length), max(test_token_length)

# 최대 sequence 길이 지정 (임의 지정)
n_seq = max(train_max_length, dev_max_length, test_max_length) + 1
print(train_max_length, dev_max_length, test_max_length, n_seq)

# pad id
pad_id = vocab.pad_id()
print('pad_id:', pad_id)

#
# 2 inputs
#

# train numpy matrix
train_labels = np.array(train_label_ids)
train_inputs_1 = np.zeros((len(train_sentence1_ids), n_seq))
train_inputs_2 = np.zeros((len(train_sentence2_ids), n_seq))
print(train_labels.shape, train_labels[:100])
print(train_inputs_1.shape, train_inputs_1[0], train_inputs_1[-1])
print(train_inputs_2.shape, train_inputs_2[0], train_inputs_2[-1])

# array test
array = [1, 2, 3, 4, 5]
print(array + [0] * 2)
print(array + [0] * 0)
print(array + [0] * -2)
print(array[:100])
print(array[:3])

# assing train_token_ids to inputs
for i, token_id in enumerate(train_sentence1_ids):
    token_id += [pad_id] * (n_seq - len(token_id))
    token_id = token_id[:n_seq]
    assert len(token_id) == n_seq
    train_inputs_1[i] = token_id
print(train_inputs_1.shape, train_inputs_1[0], train_inputs_1[-1])

for i, token_id in enumerate(train_sentence2_ids):
    token_id += [pad_id] * (n_seq - len(token_id))
    token_id = token_id[:n_seq]
    assert len(token_id) == n_seq
    train_inputs_2[i] = token_id
print(train_inputs_2.shape, train_inputs_2[0], train_inputs_2[-1])

# dev numpy matrix
dev_labels = np.array(dev_label_ids)
dev_inputs_1 = np.zeros((len(dev_sentence1_ids), n_seq))
dev_inputs_2 = np.zeros((len(dev_sentence2_ids), n_seq))
print(dev_labels.shape, dev_labels[:100])
print(dev_inputs_1.shape, dev_inputs_1[0], dev_inputs_1[-1])
print(dev_inputs_2.shape, dev_inputs_2[0], dev_inputs_2[-1])

# assing dev_token_ids to inputs
for i, token_id in enumerate(dev_sentence1_ids):
    token_id += [pad_id] * (n_seq - len(token_id))
    token_id = token_id[:n_seq]
    assert len(token_id) == n_seq
    dev_inputs_1[i] = token_id
print(dev_inputs_1.shape, dev_inputs_1[0], dev_inputs_1[-1])

for i, token_id in enumerate(dev_sentence2_ids):
    token_id += [pad_id] * (n_seq - len(token_id))
    token_id = token_id[:n_seq]
    assert len(token_id) == n_seq
    dev_inputs_2[i] = token_id
print(dev_inputs_2.shape, dev_inputs_2[0], dev_inputs_2[-1])

# test numpy matrix
test_labels = np.array(test_label_ids)
test_inputs_1 = np.zeros((len(test_sentence1_ids), n_seq))
test_inputs_2 = np.zeros((len(test_sentence2_ids), n_seq))
print(test_labels.shape, test_labels[:100])
print(test_inputs_1.shape, test_inputs_1[0], test_inputs_1[-1])
print(test_inputs_2.shape, test_inputs_2[0], test_inputs_2[-1])

# assing test_token_ids to inputs
for i, token_id in enumerate(test_sentence1_ids):
    token_id += [pad_id] * (n_seq - len(token_id))
    token_id = token_id[:n_seq]
    assert len(token_id) == n_seq
    test_inputs_1[i] = token_id
print(test_inputs_1.shape, test_inputs_1[0], test_inputs_1[-1])

for i, token_id in enumerate(test_sentence2_ids):
    token_id += [pad_id] * (n_seq - len(token_id))
    token_id = token_id[:n_seq]
    assert len(token_id) == n_seq
    test_inputs_2[i] = token_id
print(test_inputs_2.shape, test_inputs_2[0], test_inputs_2[-1])


#
# Stub model
#

def build_model_stub(n_vocab, d_model, n_output):
    """
    Stub Model
    :param n_vocab: number of vocab
    :param d_model: hidden size
    :param n_output: number of output
    :return model: model object
    """
    inputs_1 = tf.keras.layers.Input((None,))  # bs, n_seq
    inputs_2 = tf.keras.layers.Input((None,))  # bs, n_seq

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)

    hidden_1 = embedding(inputs_1)  # bs, n_seq, d_model
    hidden_2 = embedding(inputs_2)  # bs, n_seq, d_model
    max_pooling = tf.keras.layers.GlobalMaxPool1D()
    hidden_1 = max_pooling(hidden_1)  # bs, d_model
    hidden_2 = max_pooling(hidden_2)  # bs, d_model
    hidden = tf.concat([hidden_1, hidden_2], axis=-1)
    output_dense = tf.keras.layers.Dense(n_output, activation=tf.nn.softmax)
    outputs = output_dense(hidden)

    model = tf.keras.Model(inputs=(inputs_1, inputs_2), outputs=outputs)
    return model


# model build
model_stub = build_model_stub(len(vocab), 256, 3)
print(model_stub.summary())

# complie
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
model_stub.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

# train
history = model_stub.fit((train_inputs_1, train_inputs_2), train_labels, validation_data=((dev_inputs_1, dev_inputs_2), dev_labels), epochs=100, batch_size=128)


def draw_history(history):
    """
    draw training history
    :param history: training history object
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], 'b-', label='loss')
    plt.plot(history.history['val_loss'], 'r--', label='val_loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['acc'], 'g-', label='acc')
    plt.plot(history.history['val_acc'], 'k--', label='val_acc')
    plt.xlabel('Epoch')
    plt.legend()

    plt.show()


# result
draw_history(history)

#
# Evaluate
#
model_stub.evaluate((dev_inputs_1, dev_inputs_2), dev_labels)


def make_confusion_matrix_2(model, inputs_1, inputs_2, labels, n_batch=64, n_output=3):
    """
    make confusion matrix
    :param model: model object
    :param inputs_1: inputs 1
    :param inputs_2: inputs 2
    :param labels: labels
    :param n_batch: number of batch
    :param n_output: number of output
    :return confusion_matrix: confusion matrix
    """
    y_predicts = []
    for i in range(0, len(inputs_1), n_batch):
        batch_inputs_1 = inputs_1[i:i + n_batch]
        batch_inputs_2 = inputs_2[i:i + n_batch]
        batch_predict = model.predict((batch_inputs_1, batch_inputs_2))
        y_class = tf.argmax(batch_predict, axis=-1)
        y_predicts.extend(y_class)
    assert len(labels) == len(y_predicts)

    confusion_matrix = np.zeros((n_output, n_output))
    for y_true, y_pred in zip(labels, y_predicts):
        confusion_matrix[y_true, y_pred] += 1
    return confusion_matrix


# make confusion matrix
confusion_matrix = make_confusion_matrix_2(model_stub, dev_inputs_1, dev_inputs_2, test_labels)
print(confusion_matrix)


def plot_confusion_matrix(confusion_matrix, accuracy, tags):
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
    plt.colorbar()

    thresh = confusion_matrix.max() / 1.5

    tick_marks = np.arange(len(tags))
    plt.xticks(tick_marks, tags)
    plt.yticks(tick_marks, tags)

    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        total, value = np.sum(confusion_matrix[i]), confusion_matrix[i, j]
        plt.text(j, i, f'{value} ({value * 100 / max(1, total):.2f}%)',
                 horizontalalignment='center',
                 color='white' if confusion_matrix[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(f'Predicted label\naccuracy={accuracy * 100:.2f}%')
    plt.show()


# true
print(np.eye(len(confusion_matrix)))
true_matrix = np.eye(len(confusion_matrix)) * confusion_matrix
print(true_matrix, np.sum(true_matrix))

# accuracy
print(np.sum(confusion_matrix))
accuracy = np.sum(true_matrix) / np.sum(confusion_matrix)
print(accuracy)

# plot matrix
plot_confusion_matrix(confusion_matrix, accuracy, ['contradiction', 'neutral', 'entailment'])


def do_predict(vocab, model, n_seq, string1, string2):
    """
    입력에 대한 답변 생성하는 함수
    :param vocab: vocabulary object
    :param model: model object
    :param n_seq: 입력 개수
    :param string1: 입력 문자열 1
    :param string2: 입력 문자열 2
    """
    # token 생성: <string tokens>, [PAD] tokens
    token_1 = vocab.encode_as_ids(string1)
    token_1 += [0] * (n_seq - len(token_1))
    token_1 = token_1[:n_seq]

    token_2 = vocab.encode_as_ids(string2)
    token_2 += [0] * (n_seq - len(token_2))
    token_2 = token_2[:n_seq]

    y_pred = model.predict((np.array([token_1]), np.array([token_2])))
    y_pred_class = K.argmax(y_pred, axis=-1)

    return id_to_label[y_pred_class[0].numpy()]


string1 = '남자가 밥을 먹고 있다.'
string2 = '여자가 밥을 먹고 있다.'
print(f'output > {do_predict(vocab, model_stub, n_seq, string1, string2)}')


#
# Cnn
#

def build_model_cnn(n_vocab, d_model, n_output):
    """
    CNN Model
    :param n_vocab: number of vocab
    :param d_model: hidden size
    :param n_output: number of output
    :return model: model object
    """
    inputs_1 = tf.keras.layers.Input((None,))  # bs, n_seq
    inputs_2 = tf.keras.layers.Input((None,))  # bs, n_seq

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)

    hidden_1 = embedding(inputs_1)  # bs, n_seq, d_model
    conv_1d_1 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding='causal')
    hidden_1 = conv_1d_1(hidden_1)  # bs, n_seq, d_model

    hidden_2 = embedding(inputs_2)  # bs, n_seq, d_model
    conv_1d_2 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding='causal')
    hidden_2 = conv_1d_2(hidden_2)  # bs, n_seq, d_model

    max_pooling = tf.keras.layers.GlobalMaxPool1D()
    hidden_1 = max_pooling(hidden_1)  # bs, d_model
    hidden_2 = max_pooling(hidden_2)  # bs, d_model

    hidden = tf.concat([hidden_1, hidden_2], axis=-1)
    output_dense = tf.keras.layers.Dense(n_output, activation=tf.nn.softmax)
    outputs = output_dense(hidden)

    model = tf.keras.Model(inputs=(inputs_1, inputs_2), outputs=outputs)
    return model


# model build
model_cnn = build_model_cnn(len(vocab), 256, 3)
print(model_cnn.summary())

# complie
model_cnn.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

# train
history = model_stub.fit((train_inputs_1, train_inputs_2), train_labels, validation_data=((dev_inputs_1, dev_inputs_2), dev_labels), epochs=100, batch_size=128)

# result
draw_history(history)


#
# Rnn 1
#

def build_model_rnn_1(n_vocab, d_model, n_output):
    """
    RNN type1 Model
    :param n_vocab: number of vocab
    :param d_model: hidden size
    :param n_output: number of output
    :return model: model object
    """
    inputs_1 = tf.keras.layers.Input((None,))  # bs, n_seq
    inputs_2 = tf.keras.layers.Input((None,))  # bs, n_seq

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)

    hidden_1 = embedding(inputs_1)  # bs, n_seq, d_model
    rnn_1 = tf.keras.layers.SimpleRNN(units=d_model)
    hidden_1 = rnn_1(hidden_1)  # bs, d_model

    hidden_2 = embedding(inputs_2)  # bs, n_seq, d_model
    rnn_2 = tf.keras.layers.SimpleRNN(units=d_model)
    hidden_2 = rnn_1(hidden_2)  # bs, d_model

    hidden = tf.concat([hidden_1, hidden_2], axis=-1)
    output_dense = tf.keras.layers.Dense(n_output, activation=tf.nn.softmax)
    outputs = output_dense(hidden)

    model = tf.keras.Model(inputs=(inputs_1, inputs_2), outputs=outputs)
    return model


# model build
model_rnn_1 = build_model_rnn_1(len(vocab), 256, 3)
print(model_rnn_1.summary())

# complie
model_rnn_1.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

# train
history = model_stub.fit((train_inputs_1, train_inputs_2), train_labels, validation_data=((dev_inputs_1, dev_inputs_2), dev_labels), epochs=100, batch_size=128)

# result
draw_history(history)


#
# Rnn 2
#

def build_model_rnn_2(n_vocab, d_model, n_output):
    """
    RNN type1 Model
    :param n_vocab: number of vocab
    :param d_model: hidden size
    :param n_output: number of output
    :return model: model object
    """
    inputs_1 = tf.keras.layers.Input((None,))  # bs, n_seq
    inputs_2 = tf.keras.layers.Input((None,))  # bs, n_seq

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)

    hidden_1 = embedding(inputs_1)  # bs, n_seq, d_model
    rnn_1 = tf.keras.layers.SimpleRNN(units=d_model, return_sequences=True)
    hidden_1 = rnn_1(hidden_1)  # bs, n_seq, d_model

    hidden_2 = embedding(inputs_2)  # bs, n_seq, d_model
    rnn_2 = tf.keras.layers.SimpleRNN(units=d_model, return_sequences=True)
    hidden_2 = rnn_1(hidden_2)  # bs, n_seq, d_model

    max_pooling = tf.keras.layers.GlobalMaxPool1D()
    hidden_1 = max_pooling(hidden_1)  # bs, d_model
    hidden_2 = max_pooling(hidden_2)  # bs, d_model

    hidden = tf.concat([hidden_1, hidden_2], axis=-1)
    output_dense = tf.keras.layers.Dense(n_output, activation=tf.nn.softmax)
    outputs = output_dense(hidden)

    model = tf.keras.Model(inputs=(inputs_1, inputs_2), outputs=outputs)
    return model


# model build
model_rnn_2 = build_model_rnn_2(len(vocab), 256, 3)
print(model_rnn_2.summary())

# complie
model_rnn_2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

# train
history = model_stub.fit((train_inputs_1, train_inputs_2), train_labels, validation_data=((dev_inputs_1, dev_inputs_2), dev_labels), epochs=100, batch_size=128)

# result
draw_history(history)

#
# 1 inputs
#
n_seq_all = 2 * n_seq

# train numpy matrix
train_labels = np.array(train_label_ids)
train_inputs_all = np.zeros((len(train_sentence1_ids), n_seq_all))
print(train_labels.shape, train_labels[:100])
print(train_inputs_all.shape, train_inputs_all[0], train_inputs_all[-1])

# assing train_token_ids to inputs
for i, (token1_id, token2_id) in enumerate(zip(train_sentence1_ids, train_sentence2_ids)):
    token_id_all = token1_id + [vocab.piece_to_id('[SEP]')] + token2_id
    token_id_all += [pad_id] * (n_seq_all - len(token_id_all))
    token_id_all = token_id_all[:n_seq_all]
    assert len(token_id_all) == n_seq_all
    train_inputs_all[i] = token_id_all
print(train_inputs_all.shape, train_inputs_all[0], train_inputs_all[-1])

# dev numpy matrix
dev_labels = np.array(dev_label_ids)
dev_inputs_all = np.zeros((len(dev_sentence1_ids), n_seq_all))
print(dev_labels.shape, dev_labels[:100])
print(dev_inputs_all.shape, dev_inputs_all[0], dev_inputs_all[-1])

# assing dev_token_ids to inputs
for i, (token1_id, token2_id) in enumerate(zip(dev_sentence1_ids, dev_sentence2_ids)):
    token_id_all = token1_id + [vocab.piece_to_id('[SEP]')] + token2_id
    token_id_all += [pad_id] * (n_seq_all - len(token_id_all))
    token_id_all = token_id_all[:n_seq_all]
    assert len(token_id_all) == n_seq_all
    dev_inputs_all[i] = token_id_all
print(dev_inputs_all.shape, dev_inputs_all[0], dev_inputs_all[-1])

# test numpy matrix
test_labels = np.array(test_label_ids)
test_inputs_all = np.zeros((len(test_sentence1_ids), n_seq_all))
print(test_labels.shape, test_labels[:100])
print(test_inputs_all.shape, test_inputs_all[0], test_inputs_all[-1])

# assing test_token_ids to inputs
for i, (token1_id, token2_id) in enumerate(zip(test_sentence1_ids, test_sentence2_ids)):
    token_id_all = token1_id + [vocab.piece_to_id('[SEP]')] + token2_id
    token_id_all += [pad_id] * (n_seq_all - len(token_id_all))
    token_id_all = token_id_all[:n_seq_all]
    assert len(token_id_all) == n_seq_all
    test_inputs_all[i] = token_id_all
print(test_inputs_all.shape, test_inputs_all[0], test_inputs_all[-1])


#
# stub2
#

def build_model_stub_2(n_vocab, d_model, n_output):
    """
    Stub Model
    :param n_vocab: number of vocab
    :param d_model: hidden size
    :param n_output: number of output
    :return model: model object
    """
    inputs = tf.keras.layers.Input((None,))  # bs, n_seq

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)

    hidden = embedding(inputs)  # bs, n_seq, d_model
    max_pooling = tf.keras.layers.GlobalMaxPool1D()
    hidden = max_pooling(hidden)  # bs, d_model
    output_dense = tf.keras.layers.Dense(n_output, activation=tf.nn.softmax)
    outputs = output_dense(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# model build
model_stub_2 = build_model_stub_2(len(vocab), 256, 3)
print(model_stub_2.summary())

# complie
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
model_stub_2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

# train
history = model_stub_2.fit(train_inputs_all, train_labels, validation_data=(dev_inputs_all, dev_labels), epochs=100, batch_size=128)

# result
draw_history(history)


#
# Cnn
#

def build_model_cnn_2(n_vocab, d_model, n_output):
    """
    CNN Model
    :param n_vocab: number of vocab
    :param d_model: hidden size
    :param n_output: number of output
    :return model: model object
    """
    inputs = tf.keras.layers.Input((None,))  # bs, n_seq

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)

    hidden = embedding(inputs)  # bs, n_seq, d_model
    conv_1d = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding='causal')
    hidden = conv_1d(hidden)  # bs, n_seq, d_model

    max_pooling = tf.keras.layers.GlobalMaxPool1D()
    hidden = max_pooling(hidden)  # bs, d_model

    output_dense = tf.keras.layers.Dense(n_output, activation=tf.nn.softmax)
    outputs = output_dense(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# model build
model_cnn_2 = build_model_cnn_2(len(vocab), 256, 3)
print(model_cnn_2.summary())

# complie
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
model_cnn_2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

# train
history = model_cnn_2.fit(train_inputs_all, train_labels, validation_data=(dev_inputs_all, dev_labels), epochs=100, batch_size=128)

# result
draw_history(history)


#
# Rnn 1
#

def build_model_rnn_1_2(n_vocab, d_model, n_output):
    """
    RNN type1 Model
    :param n_vocab: number of vocab
    :param d_model: hidden size
    :param n_output: number of output
    :return model: model object
    """
    inputs = tf.keras.layers.Input((None,))  # bs, n_seq

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)

    hidden = embedding(inputs)  # bs, n_seq, d_model
    rnn = tf.keras.layers.SimpleRNN(units=d_model)
    hidden = rnn(hidden)  # bs, n_d_model

    output_dense = tf.keras.layers.Dense(n_output, activation=tf.nn.softmax)
    outputs = output_dense(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# model build
model_rnn_1_2 = build_model_rnn_1_2(len(vocab), 256, 3)
print(model_rnn_1_2.summary())

# complie
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
model_rnn_1_2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

# train
history = model_rnn_1_2.fit(train_inputs_all, train_labels, validation_data=(dev_inputs_all, dev_labels), epochs=100, batch_size=128)

# result
draw_history(history)


#
# Rnn 2
#

def build_model_rnn_2_2(n_vocab, d_model, n_output):
    """
    RNN type1 Model
    :param n_vocab: number of vocab
    :param d_model: hidden size
    :param n_output: number of output
    :return model: model object
    """
    inputs = tf.keras.layers.Input((None,))  # bs, n_seq

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)

    hidden = embedding(inputs)  # bs, n_seq, d_model
    rnn = tf.keras.layers.SimpleRNN(units=d_model, return_sequences=True)
    hidden = rnn(hidden)  # bs, n_seq, d_model

    max_pooling = tf.keras.layers.GlobalMaxPool1D()
    hidden = max_pooling(hidden)  # bs, d_model

    output_dense = tf.keras.layers.Dense(n_output, activation=tf.nn.softmax)
    outputs = output_dense(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# model build
model_rnn_2_2 = build_model_rnn_2_2(len(vocab), 256, 3)
print(model_rnn_2_2.summary())

# complie
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
model_rnn_2_2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

# train
history = model_rnn_2_2.fit(train_inputs_all, train_labels, validation_data=(dev_inputs_all, dev_labels), epochs=100, batch_size=128)

# result
draw_history(history)


#
# Rnn 3 reverse
#

def build_model_rnn_3_2(n_vocab, d_model, n_output):
    """
    RNN type1 Model
    :param n_vocab: number of vocab
    :param d_model: hidden size
    :param n_output: number of output
    :return model: model object
    """
    inputs = tf.keras.layers.Input((None,))  # bs, n_seq

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)

    hidden = embedding(inputs)  # bs, n_seq, d_model
    rnn = tf.keras.layers.SimpleRNN(units=d_model, return_sequences=True, go_backwards=True)
    hidden = rnn(hidden)  # bs, n_seq, d_model

    max_pooling = tf.keras.layers.GlobalMaxPool1D()
    hidden = max_pooling(hidden)  # bs, d_model

    output_dense = tf.keras.layers.Dense(n_output, activation=tf.nn.softmax)
    outputs = output_dense(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# model build
model_rnn_3_2 = build_model_rnn_3_2(len(vocab), 256, 3)
print(model_rnn_3_2.summary())

# complie
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
model_rnn_3_2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

# train
history = model_rnn_3_2.fit(train_inputs_all, train_labels, validation_data=(dev_inputs_all, dev_labels), epochs=100, batch_size=128)

# result
draw_history(history)

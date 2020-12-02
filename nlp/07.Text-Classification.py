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

nsmc_dir = os.path.join(data_dir, 'nsmc')
if not os.path.exists(nsmc_dir):
    os.makedirs(nsmc_dir)

train_txt = os.path.join(nsmc_dir, 'ratings_train.txt')
test_txt = os.path.join(nsmc_dir, 'ratings_test.txt')

#
# download file
#

wget.download('https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt', train_txt)
wget.download('https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt', test_txt)
print(os.listdir(nsmc_dir))


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
label_counts = train_data['label'].value_counts()
print(f'전체 학습 label 개수: {label_counts}')

# head=0 첫벗째 줄이 head, quoting=3 큰따옴표 무시
test_data = pd.read_csv(test_txt, header=0, delimiter='\t', quoting=3)
print(f'전체 시험 raw 개수: {len(test_data)}')
test_data = test_data.dropna()
print(f'전체 시험 valid 개수: {len(test_data)}')
test_data = test_data.sample(500)  # 빠른 확인을 위해 500개만 사용
print(f'전체 시험 sample 개수: {len(test_data)}')
label_counts = test_data['label'].value_counts()
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
train_tokens, train_labels = [], []
for i, row in train_data.iterrows():
    token = vocab.encode_as_pieces(row['document'])
    train_tokens.append(token)
    train_labels.append(row['label'])

assert len(train_tokens) == len(train_labels)

print(train_tokens[:100])
print(train_labels[:100])

test_tokens, test_labels = [], []
for i, row in test_data.iterrows():
    token = vocab.encode_as_pieces(row['document'])
    test_tokens.append(token)
    test_labels.append(row['label'])

assert len(test_tokens) == len(test_labels)

print(test_tokens[:100])
print(test_labels[:100])

#
# token to id
#
train_token_ids = [[vocab.piece_to_id(p) for p in token] for token in train_tokens]
print(train_token_ids[:100])
assert len(train_token_ids) == len(train_labels)

test_token_ids = [[vocab.piece_to_id(p) for p in token] for token in test_tokens]
print(test_token_ids[:100])
assert len(test_token_ids) == len(test_labels)

#
# pad
#

# 길이가 달라서 matrix 생성 안됨
print(np.array(train_token_ids)[:50])
print(np.array(test_token_ids)[:50])

# train token 길이 확인
train_token_length = [len(token_id) for token_id in train_token_ids]
print(train_token_length[:100])

# test token 길이 확인
test_token_length = [len(token_id) for token_id in test_token_ids]
print(test_token_length[:100])

# 최대 길이 확인
train_max_length, test_max_length = max(train_token_length), max(test_token_length)

# 최대 sequence 길이 지정 (임의 지정)
n_seq = max(train_max_length, test_max_length) - 7
print(train_max_length, test_max_length, n_seq)

# pad id
pad_id = vocab.pad_id()
print('pad_id:', pad_id)

# train numpy matrix
train_labels = np.array(train_labels)
train_inputs = np.zeros((len(train_token_ids), n_seq))
print(train_labels.shape, train_labels[:100])
print(train_inputs.shape, train_inputs[0])
print(train_inputs.shape, train_inputs[-1])

# array test
array = [1, 2, 3, 4, 5]
print(array + [0] * 2)
print(array + [0] * 0)
print(array + [0] * -2)
print(array[:100])
print(array[:3])

# assing train_token_ids to inputs
for i, token_id in enumerate(train_token_ids):
    token_id += [pad_id] * (n_seq * len(token_id))
    token_id = token_id[:n_seq]
    assert len(token_id) == n_seq
    train_inputs[i] = token_id
print(train_inputs.shape, train_inputs[0])
print(train_inputs.shape, train_inputs[-1])

# test numpy matrix
test_labels = np.array(test_labels)
test_inputs = np.zeros((len(test_token_ids), n_seq))
print(test_labels.shape, test_labels[:100])
print(test_inputs.shape, test_inputs[:100])

# assing test_token_ids to inputs
for i, token_id in enumerate(test_token_ids):
    token_id += [pad_id] * (n_seq * len(token_id))
    token_id = token_id[:n_seq]
    assert len(token_id) == n_seq
    test_inputs[i] = token_id
print(test_inputs.shape, test_inputs[0])
print(test_inputs.shape, test_inputs[-1])

# train and test data
print('train:', train_inputs.shape, train_labels.shape)
print('test:', test_inputs.shape, test_labels.shape)


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
model_stub = build_model_stub(len(vocab), 256, 2)
print(model_stub.summary())

# complie
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
model_stub.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

# train
history = model_stub.fit(train_inputs, train_labels, validation_data=(test_inputs, test_labels), epochs=100, batch_size=128)


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
model_stub.evaluate(test_inputs, test_labels)


def make_confusion_matrix(model, inputs, labels, n_batch=64, n_output=2):
    """
    make confusion matrix
    :param model: model object
    :param inputs: inputs
    :param labels: labels
    :param n_batch: number of batch
    :param n_output: number of output
    :return confusion_matrix: confusion matrix
    """
    y_predicts = []
    for i in range(0, len(inputs), n_batch):
        batch_inputs = inputs[i:i + n_batch]
        batch_predict = model.predict(batch_inputs)
        y_class = tf.argmax(batch_predict, axis=-1)
        y_predicts.extend(y_class)
    assert len(labels) == len(y_predicts)

    confusion_matrix = np.zeros((n_output, n_output))
    for y_true, y_pred in zip(labels, y_predicts):
        confusion_matrix[y_true, y_pred] += 1
    return confusion_matrix



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


# make confusion matrix
confusion_matrix = make_confusion_matrix(model_stub, test_inputs, test_labels)
print(confusion_matrix)

# true
print(np.eye(len(confusion_matrix)))
true_matrix = np.eye(len(confusion_matrix)) * confusion_matrix
print(true_matrix, np.sum(true_matrix))

# accuracy
print(np.sum(confusion_matrix))
accuracy = np.sum(true_matrix) / np.sum(confusion_matrix)
print(accuracy)

# plot matrix
plot_confusion_matrix(confusion_matrix, accuracy, ['False', 'True'])


def do_predict(vocab, model, n_seq, string):
    """
    입력에 대한 답변 생성하는 함수
    :param vocab: vocabulary object
    :param model: model object
    :param n_seq: 입력 개수
    :param string: 입력 문자열
    """
    # token 생성: <string tokens>, [PAD] tokens
    token = vocab.encode_as_ids(string)
    token += [0] * (n_seq - len(token))
    token = token[:n_seq]

    y_pred = model.predict(np.array([token]))
    y_pred_class = K.argmax(y_pred, axis=-1)

    return '긍정' if y_pred_class[0] == 1 else '부정'


string = '시간 때우기 좋아'
print(f'output > {do_predict(vocab, model_stub, n_seq, string)}')


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
model_cnn = build_model_cnn(len(vocab), 256, 2)
print(model_cnn.summary())

# complie
model_cnn.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

# train
history = model_cnn.fit(train_inputs, train_labels, validation_data=(test_inputs, test_labels), epochs=100, batch_size=128)

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
    inputs = tf.keras.layers.Input((None,))  # bs, n_seq

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)
    hidden = embedding(inputs)  # bs, n_seq, d_model
    rnn = tf.keras.layers.SimpleRNN(units=d_model)
    hidden = rnn(hidden)  # bs, d_model
    output_dense = tf.keras.layers.Dense(n_output, activation=tf.nn.softmax)
    outputs = output_dense(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# model build
model_rnn_1 = build_model_rnn_1(len(vocab), 256, 2)
print(model_rnn_1.summary())

# complie
model_rnn_1.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

# train
history = model_rnn_1.fit(train_inputs, train_labels, validation_data=(test_inputs, test_labels), epochs=100, batch_size=128)

# result
draw_history(history)


#
# Rnn 2
#

def build_model_rnn_2(n_vocab, d_model, n_output):
    """
    RNN type2 Model
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
model_rnn_2 = build_model_rnn_2(len(vocab), 256, 2)
print(model_rnn_2.summary())

# complie
model_rnn_2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

# train
history = model_rnn_2.fit(train_inputs, train_labels, validation_data=(test_inputs, test_labels), epochs=100, batch_size=128)

# result
draw_history(history)

#
# RNN reverse order input
#

print(train_inputs[0], train_inputs[-1])
print(test_inputs[0], test_inputs[-1])

# flip
train_flip_inputs = np.flip(train_inputs, axis=-1)
test_flip_inputs = np.flip(test_inputs, axis=-1)
print(train_flip_inputs[0], train_flip_inputs[-1])
print(test_flip_inputs[0], test_flip_inputs[-1])

# model build
model_rnn_2 = build_model_rnn_2(len(vocab), 256, 2)
print(model_rnn_2.summary())

# complie
model_rnn_2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

# train by flip_inputs
history = model_rnn_2.fit(train_flip_inputs, train_labels, validation_data=(test_flip_inputs, test_labels), epochs=100, batch_size=128)

# result
draw_history(history)


#
# RNN reverse order (go_backwards)
#

def build_model_rnn_3(n_vocab, d_model, n_output):
    """
    RNN type3 Model go_backwards
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
model_rnn_3 = build_model_rnn_3(len(vocab), 256, 2)
print(model_rnn_3.summary())

# complie
model_rnn_3.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

# train by flip_inputs
history = model_rnn_3.fit(train_inputs, train_labels, validation_data=(test_inputs, test_labels), epochs=100, batch_size=128)

# result
draw_history(history)

# -*- coding:utf-8 -*-
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

songys_dir = os.path.join(data_dir, 'songys')
if not os.path.exists(songys_dir):
    os.makedirs(songys_dir)

train_txt = os.path.join(songys_dir, 'ChatbotData.csv')


#
# file check
#

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


#
# download file
#

wget.download('https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData .csv', train_txt)
print(os.listdir(songys_dir))

print_file(train_txt)

#
# data read
# https://pandas.pydata.org/pandas-docs/stable/index.html
#

# head=0 첫벗째 줄이 head
train_data = pd.read_csv(train_txt, header=0, delimiter=',')
print(f'전체 학습 raw 개수: {len(train_data)}')
train_data = train_data.dropna()
print(f'전체 학습 valid 개수: {len(train_data)}')
train_data = train_data.sample(1000)  # 빠른 확인을 위해 1000개만 사용
print(f'전체 학습 sample 개수: {len(train_data)}')
label_counts = train_data['label'].value_counts()
print(f'전체 학습 label 개수: {label_counts}')

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
questions, answers = [], []
for i, row in train_data.iterrows():
    question = vocab.encode_as_pieces(row['Q'])
    questions.append(question)
    answer = vocab.encode_as_pieces(row['A'])
    answers.append(answer)

assert len(questions) == len(answers)

print(questions[:100])
print(answers[:100])

#
# token to id
#

question_ids = [[vocab.piece_to_id(p) for p in question] for question in questions]
answer_ids = [[vocab.piece_to_id(p) for p in answer] for answer in answers]
print(question_ids[:100])
print(answer_ids[:100])

#
# pad
#

# 길이가 달라서 matrix 생성 안됨
print(np.array(question_ids)[:50])
print(np.array(answer_ids)[:50])

# 길이 확인
question_length = [len(question_id) for question_id in question_ids]
print(question_length[:100])
answer_length = [len(answer_id) for answer_id in answer_ids]
print(answer_length[:100])

# 최대 길이 확인
answer_max_length, question_max_length = max(question_length), max(answer_length)

# 최대 sequence 길이 지정 (임의 지정)
n_seq = max(answer_max_length, question_max_length) + 2
print(answer_max_length, question_max_length, n_seq)

# pad id
pad_id = vocab.pad_id()
print('pad_id:', pad_id)

#
# inputs
#

# train numpy matrix
enc_inputs = np.zeros((len(question_ids), n_seq))
dec_inputs = np.zeros((len(answer_ids), n_seq))
dec_labels = np.zeros((len(answer_ids), n_seq))

print(enc_inputs.shape, enc_inputs[0], enc_inputs[-1])
print(dec_inputs.shape, dec_inputs[0], dec_inputs[-1])
print(dec_labels.shape, dec_labels[0], dec_labels[-1])

# assing question_ids to enc_inputs
for i, token_id in enumerate(question_ids):
    token_id += [0] * (n_seq - len(token_id))
    token_id = token_id[:n_seq]
    assert len(token_id) == n_seq
    enc_inputs[i] = token_id
print(enc_inputs.shape, enc_inputs[0], enc_inputs[-1])

# assing answer_ids to dec_inputs and dec_labels
n_max = n_seq - 1
for i, token_id in enumerate(answer_ids):
    token_id = token_id[:n_max]

    dec_input = [vocab.bos_id()] + token_id
    dec_input += [0] * (n_seq - len(dec_input))

    dec_label = token_id + [vocab.eos_id()]
    dec_label += [0] * (n_seq - len(dec_label))
    assert len(dec_input) == len(dec_label) == n_seq

    dec_inputs[i] = dec_input
    dec_labels[i] = dec_label

print(dec_inputs.shape, dec_inputs[0].astype(np.int), dec_inputs[-1].astype(np.int))
print(dec_labels.shape, dec_labels[0].astype(np.int), dec_labels[-1].astype(np.int))

train_inputs = (enc_inputs, dec_inputs)


#
# loss and accuracy
#

def lm_loss(y_true, y_pred):
    """
    pad 부분을 제외하고 loss를 계산하는 함수
    :param y_true: 정답
    :param y_pred: 예측 값
    :retrun loss: pad 부분이 제외된 loss 값
    """
    loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss *= mask
    return loss


def lm_acc(y_true, y_pred):
    """
    pad 부분을 제외하고 accuracy를 계산하는 함수
    :param y_true: 정답
    :param y_pred: 예측 값
    :retrun loss: pad 부분이 제외된 accuracy 값
    """
    y_pred_class = tf.cast(K.argmax(y_pred, axis=-1), tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    matches = tf.cast(K.equal(y_true, y_pred_class), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    matches *= mask
    accuracy = K.sum(matches) / K.maximum(K.sum(mask), 1)
    return accuracy


#
# rnn
#
def build_model_rnn(n_vocab, d_model):
    """
    rnn model build
    :param n_vocab: number of vocab
    :param d_model: hidden size
    :return model: model
    """
    enc_inputs = tf.keras.layers.Input((None,))
    dec_inputs = tf.keras.layers.Input((None,))

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)

    enc_hidden = embedding(enc_inputs)  # bs, n_seq, d_model
    enc_hidden, fw_h = tf.keras.layers.SimpleRNN(units=d_model, return_state=True)(enc_hidden)  # bs, d_model

    dec_hidden = embedding(dec_inputs)  # bs, n_seq, d_model
    dec_hidden = tf.keras.layers.SimpleRNN(units=d_model, return_sequences=True)(dec_hidden, initial_state=[fw_h])  # bs, n_seq, d_model

    outputs = tf.keras.layers.Dense(n_vocab, activation=tf.nn.softmax)(dec_hidden)

    model = tf.keras.Model(inputs=(enc_inputs, dec_inputs), outputs=outputs)
    return model


# model build
model_rnn = build_model_rnn(len(vocab), 256)
print(model_rnn.summary())

# complie
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
model_rnn.compile(loss=lm_loss, optimizer=tf.keras.optimizers.Adam(), metrics=[lm_acc])

# early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='lm_acc', patience=10)
# save weights
save_rnn_file = os.path.join(songys_dir, 'rnn.hdf5')
save_weights = tf.keras.callbacks.ModelCheckpoint(save_rnn_file, monitor='lm_acc', verbose=1, save_best_only=True, mode='max', save_freq='epoch', save_weights_only=True)

# train
history = model_rnn.fit(train_inputs, dec_labels, epochs=500, batch_size=128, callbacks=[early_stopping, save_weights])


def draw_history(history, acc='lm_acc'):
    """
    draw training history
    :param history: training history object
    :param acc: acc key
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], 'b-', label='loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history[acc], 'g-', label=acc)
    plt.xlabel('Epoch')
    plt.legend()

    plt.show()


draw_history(history)

#
# chat inference
#
string = '안녕 만나서 반가워'
enc_tokens = vocab.encode_as_pieces(string)
print(enc_tokens)
enc_token_id = [vocab.piece_to_id(p) for p in enc_tokens][:n_max]
print(enc_token_id)
enc_inputs = enc_token_id
enc_inputs += [0] * (n_seq - len(enc_inputs))
assert len(enc_inputs) == n_seq
print(enc_inputs)

dec_inputs = [vocab.bos_id()]
dec_inputs += [0] * (n_seq - len(dec_inputs))
assert len(dec_inputs) == n_seq
print(dec_inputs)

results = []
print(results)

result = model_rnn.predict((np.array([enc_inputs]), np.array([dec_inputs])))
prob = result[0][0]
print(prob.shape, prob)
word_id = int(np.random.choice(len(vocab), 1, p=prob)[0])
print(word_id)

results.append(word_id)
dec_inputs[1] = word_id
print(dec_inputs)
result = model_rnn.predict((np.array([enc_inputs]), np.array([dec_inputs])))
prob = result[0][1]
print(prob.shape, prob)
word_id = int(np.random.choice(len(vocab), 1, p=prob)[0])
print(word_id)

results.append(word_id)
dec_inputs[2] = word_id
print(dec_inputs)
result = model_rnn.predict((np.array([enc_inputs]), np.array([dec_inputs])))
prob = result[0][2]
print(prob.shape, prob)
word_id = int(np.random.choice(len(vocab), 1, p=prob)[0])
print(word_id)


def do_predict(vocab, model, n_seq, string):
    """
    응답을 순차적으로 생성
    :param vocab: vocab
    :param model: model object
    :param n_seq: 시퀀스 길이 (number of sequence)
    :param string: 입력 문자열
    :return response: 입력 문자열에 대한 응답
    """
    # encoder_tokens = vocab.encode_as_pieces(string)
    enc_inputs = vocab.encode_as_ids(string)[:n_seq]
    enc_inputs += [0] * (n_seq - len(enc_inputs))
    assert len(enc_inputs) == n_seq

    # decoder_tokens = ['[BOS]']
    dec_inputs = [vocab.bos_id()]
    dec_inputs += [0] * (n_seq - len(dec_inputs))

    response = []
    for i in range(n_seq - 1):
        outputs = model.predict([np.array([enc_inputs]), np.array([dec_inputs])])
        prob = outputs[0][i]
        word_id = int(np.random.choice(len(vocab), 1, p=prob)[0])
        if word_id == vocab.eos_id():
            break
        response.append(word_id)
        dec_inputs[i + 1] = word_id
    return vocab.decode_ids(response)


model_rnn = build_model_rnn(len(vocab), 256)
print(model_rnn.summary())

string = '안녕 만나서 반가워'
print(do_predict(vocab, model_rnn, n_seq, string))

model_rnn.load_weights(save_rnn_file)

print(do_predict(vocab, model_rnn, n_seq, string))


#
# bi rnn
#
def build_model_bi_rnn(n_vocab, d_model):
    """
    bi rnn model build
    :param n_vocab: number of vocab
    :param d_model: hidden size
    :return model: model
    """
    enc_inputs = tf.keras.layers.Input((None,))
    dec_inputs = tf.keras.layers.Input((None,))

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)

    enc_hidden = embedding(enc_inputs)  # bs, n_seq, d_model
    enc_hidden, fw_h, bw_h = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=d_model, return_state=True))(enc_hidden)  # bs, 2 * d_model
    s_h = tf.concat([fw_h, bw_h], axis=-1)  # bs, 2 * d_model

    dec_hidden = embedding(dec_inputs)  # bs, n_seq, d_model
    dec_hidden = tf.keras.layers.SimpleRNN(units=d_model * 2, return_sequences=True)(dec_hidden, initial_state=[s_h])  # bs, n_seq, 2 * d_model

    outputs = tf.keras.layers.Dense(n_vocab, activation=tf.nn.softmax)(dec_hidden)

    model = tf.keras.Model(inputs=(enc_inputs, dec_inputs), outputs=outputs)
    return model


# model build
model_bi_rnn = build_model_bi_rnn(len(vocab), 256)
print(model_bi_rnn.summary())

# complie
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
model_bi_rnn.compile(loss=lm_loss, optimizer=tf.keras.optimizers.Adam(), metrics=[lm_acc])

# early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='lm_acc', patience=10)
# save weights
save_bi_rnn_file = os.path.join(songys_dir, 'bi_rnn.hdf5')
save_weights = tf.keras.callbacks.ModelCheckpoint(save_bi_rnn_file, monitor='lm_acc', verbose=1, save_best_only=True, mode='max', save_freq='epoch', save_weights_only=True)

# train
history = model_bi_rnn.fit(train_inputs, dec_labels, epochs=500, batch_size=128, callbacks=[early_stopping, save_weights])

# history
draw_history(history)

model_bi_rnn = build_model_bi_rnn(len(vocab), 256)
print(model_bi_rnn.summary())

print(do_predict(vocab, model_rnn, n_seq, string))

model_bi_rnn.load_weights(save_bi_rnn_file)

print(do_predict(vocab, model_bi_rnn, n_seq, string))

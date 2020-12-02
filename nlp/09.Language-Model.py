# -*- coding:utf-8 -*-
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
import tensorflow as tf
import tensorflow.keras.backend as K

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

kowiki_dir = os.path.join(data_dir, 'kowiki')

train_txt = os.path.join(kowiki_dir, 'kowiki.txt')


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


print_file(train_txt, count=30)

#
# data read
# https://pandas.pydata.org/pandas-docs/stable/index.html
#

docs = []
with open(train_txt) as f:
    doc = []
    for i, line in enumerate(f):
        line = line.strip()
        if line:
            doc.append(line)
        elif doc:
            docs.append(doc)
            doc = []
        if 30 <= len(docs):
            break

print(len(docs))
print(docs[0])
print(docs[1])
print(docs[-1])

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

doc_tokens = [[vocab.encode_as_pieces(line) for line in doc] for doc in docs]

print(len(doc_tokens))
print(doc_tokens[0])
print(doc_tokens[1])
print(doc_tokens[-1])

n_seq = 128
n_max = n_seq - 1  # [BOS] tokens or tokens [EOS]
train_tokens = []

for doc in doc_tokens:
    chunk = []
    for i, line in enumerate(doc):
        chunk.extend(line)
        if n_max <= len(chunk) or i == len(doc) - 1:
            train_tokens.append(chunk)
            chunk = []

print(len(train_tokens))
print(train_tokens[0])
print(train_tokens[1])
print(train_tokens[-1])

#
# token to id
#
train_token_ids = []
for train_token in train_tokens:
    token_id = [vocab.piece_to_id(p) for p in train_token]
    train_token_ids.append(token_id)

print(len(train_token_ids))
print(train_token_ids[0])
print(train_token_ids[1])
print(train_token_ids[-1])

#
# inputs
#

# 길이가 달라서 matrix 생성 안됨
print(np.array(train_token_ids)[:50])

train_inputs = np.zeros((len(train_token_ids), n_seq))
train_labels = np.zeros((len(train_token_ids), n_seq))
print(train_inputs.shape, train_inputs[0], train_inputs[-1])
print(train_labels.shape, train_labels[0], train_labels[-1])

for i, token_id in enumerate(train_token_ids):
    token_id = token_id[:n_max]

    input_id = [vocab.bos_id()] + token_id
    input_id += [0] * (n_seq - len(input_id))

    label_id = token_id + [vocab.eos_id()]
    label_id += [0] * (n_seq - len(label_id))

    assert len(input_id) == len(label_id) == n_seq
    train_inputs[i] = input_id
    train_labels[i] = label_id

print(train_inputs.shape, train_labels.shape)
print(train_inputs[0].astype(np.int32))
print(train_labels[0].astype(np.int32))
print(train_inputs[-1].astype(np.int32))
print(train_labels[-1].astype(np.int32))


#
# model
#
def build_model_rnn(n_vocab, d_model):
    """
    rnn model build
    :param n_vocab: number of vocab
    :param d_model: hidden size
    :return model: model
    """
    inputs = tf.keras.layers.Input((None,))

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)

    hidden = embedding(inputs)
    hidden = tf.keras.layers.SimpleRNN(units=d_model, return_sequences=True)(hidden)
    outputs = tf.keras.layers.Dense(n_vocab, activation=tf.nn.softmax)(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# model build
model_rnn = build_model_rnn(len(vocab), 256)
print(model_rnn.summary())

# compile
model_rnn.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

# train
history = model_rnn.fit(train_inputs, train_labels, epochs=300, batch_size=512)


def draw_history(history, acc='acc'):
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
# loss and accuracy
#

# count of zero
pad_labels = (train_labels == 0)
print(pad_labels[-1])
pad_labels = pad_labels.astype(np.int)
print(pad_labels.shape)
total = pad_labels.shape[0] * pad_labels.shape[1]
n_pad = np.sum(pad_labels)
print(total, n_pad, n_pad / total)

# test value
y_true = np.random.randint(0, 10, (2, 32)).astype(np.float32)
print(y_true)
y_pred = np.random.rand(2, 32, 10).astype(np.float32)
y_pred = tf.nn.softmax(y_pred, axis=-1)  # 확률 값으로 변경
print(y_pred)

# compute loss
loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
print(loss.shape, loss)

# make mask
mask = tf.not_equal(y_true, 0)
print(mask)
mask = tf.cast(mask, tf.float32)
print(y_true)
print(mask)

# mask loss
loss *= mask
print(loss)


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


print(lm_loss(y_true, y_pred))

# compute accuracy
y_pred_class = K.argmax(y_pred, axis=-1)
print(y_pred_class)
y_pred_class = tf.cast(y_pred_class, tf.float32)
print(y_pred_class)
y_true = tf.cast(y_true, tf.float32)
matches = K.equal(y_true, y_pred_class)
print(matches)
matches = tf.cast(matches, tf.float32)
print(matches)

# make mask
mask = tf.not_equal(y_true, 0)
print(mask)
mask = tf.cast(mask, tf.float32)
print(y_true)
print(mask)
print(matches)

# mask matchs
matches *= mask
print(matches)

# accuracy
accuracy = K.sum(matches) / K.maximum(K.sum(mask), 1)  # 분모의 1을 0 방지
print(accuracy)


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


print(lm_acc(y_true, y_pred))

#
# train with lm_loss and lm_acc
#

# model build
model_rnn = build_model_rnn(len(vocab), 256)
print(model_rnn.summary())

# compile
model_rnn.compile(loss=lm_loss, optimizer=tf.keras.optimizers.Adam(), metrics=[lm_acc])

# train
history = model_rnn.fit(train_inputs, train_labels, epochs=300, batch_size=512)

draw_history(history, acc='lm_acc')

#
# text generation
#

string = '안녕'
tokens = vocab.encode_as_pieces(string)
print(tokens)
token_id = [vocab.piece_to_id(p) for p in tokens][:n_max]
print(token_id)
inputs = [vocab.bos_id()] + token_id
inputs += [0] * (n_seq - len(inputs))
assert len(inputs) == n_seq
print(inputs)
results = []
results.extend(token_id)
print(results)

result = model_rnn.predict(np.array([inputs]))
prob = result[0][1]
print(prob.shape, prob)
word_id = int(np.random.choice(len(vocab), 1, p=prob)[0])
print(word_id)

results.append(word_id)
inputs[2] = word_id
print(inputs)
result = model_rnn.predict(np.array([inputs]))
prob = result[0][2]
print(prob.shape, prob)
word_id = int(np.random.choice(len(vocab), 1, p=prob)[0])
print(word_id)

results.append(word_id)
inputs[3] = word_id
print(inputs)
result = model_rnn.predict(np.array([inputs]))
prob = result[0][3]
print(prob.shape, prob)
word_id = int(np.random.choice(len(vocab), 1, p=prob)[0])
print(word_id)


def do_generate(vocab, model, n_seq, string):
    """
    LM language generate
    :param vocab: vocab
    :param model: model
    :param n_seq: number of seqence
    :param string: inpust string
    """
    tokens = vocab.encode_as_pieces(string)
    token_ids = vocab.encode_as_ids(string)[:n_seq - 1]
    inputs = [vocab.bos_id()] + token_ids
    inputs += [0] * (n_seq - len(inputs))
    print(inputs)

    response = []
    response.extend(token_ids)
    for i in range(len(token_ids), n_seq - 1):
        outputs = model.predict(np.array([inputs]))
        prob = outputs[0][i]
        word_id = int(np.random.choice(len(vocab), 1, p=prob)[0])
        if word_id == vocab.eos_id():
            break
        response.append(word_id)
        inputs[i + 1] = word_id
    return vocab.decode_ids(response)


string = '안녕'
print(do_generate(vocab, model_rnn, n_seq, string))

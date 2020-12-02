# -*- coding:utf-8 -*-

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
# https://bcho.tistory.com/1210


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

#
# 말뭉치
#
corpus = """수학은 수식이 복잡해서 어렵다
수학은 공식이 많아서 어렵다
수학은 수식이 이해되면 쉽다
수학은 공식이 능통하면 쉽다
영어는 단어가 많아서 어렵다
영어는 듣기가 복잡해서 어렵다
영어는 단어가 이해되면 쉽다
영어는 듣기가 능통하면 쉽다
국어는 지문이 복잡해서 어렵다
국어는 한문이 많아서 어렵다
국어는 지문이 이해되면 쉽다
국어는 한문이 능통하면 쉽다"""

#
# Word Vocabulary
#

# unique words
words = list(dict.fromkeys(corpus.split()))
print(len(words), ':', words)

# word to id
word_to_id = {'[PAD]': 0, '[UNK]': 1}  # PAD: token 수가 다를경우 길이를 맞춰 줌, UNK: vocab에 없는 token
for word in words:
    word_to_id[word] = len(word_to_id)
print(len(word_to_id), ':', word_to_id)

# id to word
id_to_word = {i: word for word, i in word_to_id.items()}
print(len(word_to_id), ':', id_to_word)

# number of vocab
assert len(word_to_id) == len(id_to_word)
n_word_vocab = len(word_to_id)

#
# Tokenize
#

# split by line
lines = corpus.split('\n')
print(lines)

# tokenize
tokens = [line.split() for line in lines]
print(tokens)

#
# 전처리
#

# center-outer 생성
window_size = 2
word_pairs = []
for line_token in tokens:
    for i in range(len(line_token)):
        o_1 = max(0, i - window_size)
        o_2 = min(len(line_token) - 1, i + window_size)
        c = line_token[i]
        word_pair = {'c': c, 'o': [line_token[j] for j in range(o_1, o_2 + 1) if j != i]}
        word_pairs.append(word_pair)
print(f'word_pairs : {len(word_pairs)}')
for word_pair in word_pairs:
    print(word_pair)


#
# draw 함수
#

def plt_embdeeding(word_to_id, embedding, min_index=2):
    """
    embedding plot 표현
    :param word_to_id: word to id vocab
    :param embedding: embedding object
    :min_index: display min index (특수 문자 제거용)
    """
    # font_name = fm.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()
    # font_name = 'AppleGothic'
    font_name = 'NanumBarunGothic'

    plt.figure(figsize=(8, 8))
    plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False  # 한글 폰트 사용시 - 깨지는 문제 해결

    for label, i in word_to_id.items():
        if i < min_index:
            continue
        value = embedding(i).numpy()
        plt.scatter(value[0], value[1])
        plt.annotate(label, xy=(value[0], value[1]), xytext=(6, 4), textcoords='offset points', ha='right', va='bottom')
    plt.show()


#
# Skip Gram
#

# skip gram dataset 생성
tokens = []
labels = []
for word_pair in word_pairs:
    c = word_pair['c']
    o = word_pair['o']
    for w in o:
        tokens.append(c)
        labels.append(w)
print(f'tokens : {len(tokens)}')
print(f'tokens : {tokens}')
print(f'labels : {len(labels)}')
print(f'labels : {labels}')

# token to id
token_ids = np.array([word_to_id[token] for token in tokens])
print(f'token_ids : {len(token_ids)}')
print(f'token_ids : {token_ids}')

# label to id
label_ids = np.array([word_to_id[label] for label in labels])
print(f'label_ids : {len(label_ids)}')
print(f'label_ids : {label_ids}')

#
# modeling tutorial

# hyper parameter
n_vocab = len(word_to_id)
d_model = 4

# inputs
tmp_inputs = token_ids[:5]
print(tmp_inputs)

# embedding lookup
tmp_embedding = tf.keras.layers.Embedding(len(word_to_id), d_model)
tmp_hidden = tmp_embedding(tmp_inputs)
print(tmp_hidden)

# vocab predict
tmp_output = tf.keras.layers.Dense(n_vocab, activation=tf.nn.softmax)(tmp_hidden)
print(tmp_output)


#
# modeling

def build_model_skipgram(n_vocab, d_model):
    """
    skipgram model build
    :param n_vocab: number of vocab
    :param d_model: hidden size
    :return model: model object
    :return embedding: embedding object (화면 출력 용)
    """
    input = tf.keras.layers.Input(shape=(1,))

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)  # (n_vocab x d_model)
    hidden = embedding(input)  # (bs, 1, d_model)
    output = tf.keras.layers.Dense(n_vocab, activation=tf.nn.softmax)(hidden)  # (bs, 1, n_vocab)

    model = tf.keras.Model(inputs=input, outputs=output)
    return model, embedding


# hyper parameter
n_vocab = len(word_to_id)
d_model = 2

# model build and compile
model, embedding = build_model_skipgram(n_vocab, d_model)
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
print(model.summary())

# embedding 출력
print(embedding.get_weights())

# training 최초 출력
print(f'before training >>>')
plt_embdeeding(word_to_id, embedding)
epoch = 500

# 500회 학습 후 출력 10회 반복
for i in range(10):
    model.fit([token_ids], label_ids, batch_size=120, epochs=epoch, verbose=0)
    print(f'training >>> {(i + 1) * epoch}')
    plt_embdeeding(word_to_id, embedding)

# embedding 출력
print(embedding.get_weights())

#
# CBOW
#

# cobw dataset 생성
tokens = []
labels = []
for word_pair in word_pairs:
    c = word_pair['c']
    o = word_pair['o']
    o += ['[PAD]'] * (window_size * 2 - len(o))
    tokens.append(o)
    labels.append(c)
print(f'tokens : {len(tokens)}')
print(f'tokens : {tokens}')
print(f'labels : {len(labels)}')
print(f'labels : {labels}')

# token to id
token_ids = np.array([[word_to_id[token] for token in token_line] for token_line in tokens])
print(f'token_ids : {len(token_ids)}')
print(f'token_ids : {token_ids}')

# label to id
label_ids = np.array([word_to_id[label] for label in labels])
print(f'label_ids : {len(label_ids)}')
print(f'label_ids : {label_ids}')

#
# modeling tutorial

# hyper parameter
n_vocab = len(word_to_id)
d_model = 4

# inputs
tmp_inputs = token_ids[:5]
print(tmp_inputs)

# embedding lookup
tmp_embedding = tf.keras.layers.Embedding(len(word_to_id), d_model)
tmp_hidden_1 = tmp_embedding(tmp_inputs)
print(tmp_hidden_1)

# mean value
tmp_hidden_2 = tf.reduce_mean(tmp_hidden_1, axis=1)
print(tmp_hidden_2)

# vocab predict
tmp_output = tf.keras.layers.Dense(n_vocab, activation=tf.nn.softmax)(tmp_hidden)
print(tmp_output)


def build_model_cobw(n_vocab, d_model, n_seq):
    """
    cobw model build
    :param n_vocab: number of vocab
    :param d_model: hidden size
    :param n_seq: number of sequence (2 * window)
    :return model: model object
    :return embedding: embedding object (화면 출력 용)
    """
    input = tf.keras.layers.Input(shape=(n_seq,))

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)  # (n_vocab x d_model)
    hidden_1 = embedding(input)  # (bs, n_seq, d_model)
    hidden_2 = tf.reduce_mean(hidden_1, axis=-1)
    output = tf.keras.layers.Dense(n_vocab, activation=tf.nn.softmax)(hidden_2)  # (bs, 1, n_vocab)

    model = tf.keras.Model(inputs=input, outputs=output)
    return model, embedding


# hyper parameter
n_vocab = len(word_to_id)
d_model = 2

# model build and compile
model, embedding = build_model_cobw(n_vocab, d_model, window_size * 2)
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
print(model.summary())

# embedding 출력
print(embedding.get_weights())

# training 최초 출력
print(f'before training >>>')
plt_embdeeding(word_to_id, embedding)
epoch = 1000  # 학습 데이터 수가 작아서 2배 학습

# 500회 학습 후 출력 10회 반복
for i in range(10):
    model.fit([token_ids], label_ids, batch_size=120, epochs=epoch, verbose=0)
    print(f'training >>> {(i + 1) * epoch}')
    plt_embdeeding(word_to_id, embedding)

# embedding 출력
print(embedding.get_weights())

#
# Gensim Usage
# https://radimrehurek.com/gensim/models/word2vec.html
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-auto-examples-tutorials-run-word2vec-py
# https://radimrehurek.com/gensim/models/keyedvectors.html?highlight=word2veckeyedvectors#gensim.models.keyedvectors.KeyedVectors
#

# pip install gensim

import gensim
import gensim.downloader as api

# data_dir 선언
data_dir = './data'
if not os.path.exists(data_dir):
    data_dir = '../data'
print(os.listdir(data_dir))

gensim_dir = os.path.join(data_dir, 'gensim')
if not os.path.exists(gensim_dir):
    os.makedirs(gensim_dir)

# model download
# wv = api.load('word2vec-google-news-300') # 1.6G
wv = api.load('glove-wiki-gigaword-100')  # 128M
print(type(wv))

# model vocab print first 20
print(f'len: {len(wv.vocab)}')
for i, word in enumerate(wv.vocab):
    if i >= 20:
        break
    print(f'{i:2d}: {word}')

# king - man + woman by
# 3CosAdd being a linear sum, allows one large similarity term to dominate the expression.
# It ignores that each term reflects a different aspect of similarity, and the different aspects have different scales.
result = wv.most_similar(positive=['woman', 'king'], negative=['man'])
print(f'result: {result}')

# king - man + woman by
# 3CosMul amplifies the differences between small quantities and reduces the differences between larger ones.
result = wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
print(f'result: {result}')

# dosent match (가장 일치하지 않는 것)
result = wv.doesnt_match('breakfast basketball dinner lunch'.split())
print(f'result: {result}')

# woman & man similarity
result = wv.similarity('woman', 'man')
print(f'result: {result}')

# car similar word
result = wv.similar_by_word('cat')
print(f'result: {result}')

# distance of same word
result = wv.distance('media', 'media')
print(f'result: {result:.5f}')

#
# Gensim Training
#
# wget http://mattmahoney.net/dc/text8.zip
# unzip text8.zip

sentences = gensim.models.word2vec.Text8Corpus(os.path.join(gensim_dir, 'text8'))
for i, sentence in enumerate(sentences):
    if 10 < i:
        break
    print(sentence)

# model training (5분 정도 소요)
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=5)
# 학습된 모델
model.save(os.path.join(gensim_dir, 'text8.w2v'))

# model vocab print first 20
print(f'len: {len(model.wv.vocab)}')
for i, word in enumerate(model.wv.vocab):
    if i >= 20:
        break
    print(f'{i:2d}: {word}')

# 학습된 모델로 king - man + woman by 3CosAdd
result = model.most_similar(positive=['woman', 'king'], negative=['man'])
print(f'result: {result}')

# 학습된 모델로 king - man + woman by 3CosMul
result = model.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
print(f'result: {result}')

# TSNE를 이용해 2차원으로 차원 축소
vectors = []  # voectors
labels = []  # words
for i, word in enumerate(model.wv.vocab):
    if 3000 < i:  # 3000개만 표현
        break
    vectors.append(model.wv[word])
    labels.append(word)

vectors = np.asarray(vectors)
tsne = TSNE(n_components=2, random_state=0)
vectors = tsne.fit_transform(vectors)

x_vals = [v[0] for v in vectors]
y_vals = [v[1] for v in vectors]

print(f'labels: {labels[:30]}')
print(f'x_vals: {x_vals[:30]}')
print(f'y_vals: {y_vals[:30]}')

# 그래프 출력
plt.figure(figsize=(12, 12))
plt.scatter(x_vals, y_vals)

indices = list(range(len(labels)))
# 랜덤하게 25개만 label 출력
selected_indices = random.sample(indices, 25)
for i in selected_indices:
    plt.annotate(labels[i], (x_vals[i], y_vals[i]))
plt.show()

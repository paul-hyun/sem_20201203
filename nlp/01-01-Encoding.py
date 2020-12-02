# -*- coding:utf-8 -*-

import os

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#
# 말뭉치
#
corpus = """나는 책을 샀다
나는 책을 본다
나는 책을 팔았다
너는 책을 샀다
너는 책을 본다
너는 책을 팔았다
나는 책을 서점에서 샀다
나는 책을 도서관에서 본다
나는 책을 책방에 팔았다 너는 책을 도서관에서 본다
너는 책을 도서관에서 본다 너는 책을 서점에서 샀다"""

#
# Vocabulary
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
print(len(id_to_word), ':', id_to_word)

# number of vocab
assert len(word_to_id) == len(word_to_id)
n_vocab = len(word_to_id)

#
# Tokenize
#

# split by line
lines = corpus.split('\n')
print(lines)

# tokenize
tokens = [line.split() for line in lines]
print(tokens)

# token to id
word_ids = [[word_to_id[token] for token in line] for line in tokens]
print(word_ids)

# validate
id_words = [[id_to_word[i] for i in line] for line in word_ids]
print(id_words)

#
# OneHot
#

# one hot encoding
one_hots = []
for line_token in word_ids:
    line_one_hot = []
    for id in line_token:
        one_hot = [0] * n_vocab
        one_hot[id] = 1
        line_one_hot.append(one_hot)
    one_hots.append(line_one_hot)
print(one_hots)

# one hot encoding by tf (fail non-rectangular)
try:
    print(np.array(word_ids))  # matrix 가 만들어 지지 않음
    tf_one_hots = tf.one_hot(indices=word_ids, depth=n_vocab)
    print(tf_one_hots)
except Exception as e:
    print(e)

# 데이터 길이 확인
lenghts = [len(line_token) for line_token in word_ids]
max_len = max(lenghts)
print(max_len, ':', lenghts)

# 데이터 길이 맞춤
token_align_ids = []
for line_token in word_ids:
    aligned_line_token = line_token + [0] * (max_len - len(line_token))
    token_align_ids.append(aligned_line_token)

# 길이 확인
lenghts = [len(line_token) for line_token in token_align_ids]
print(lenghts)

# 다시 one hot encoding by tf
tf_one_hots = tf.one_hot(indices=np.array(token_align_ids), depth=n_vocab)
print(tf_one_hots)

#
# Embedding
#

# numpy embedding
d_model = 4  # hidden length
np_embedding = np.random.randint(1, 100, (n_vocab, d_model)) / 100
print(np_embedding)

# one hot matmul
token_idx = 5
one_hot = [0.] * n_vocab
one_hot[token_idx] = 1.
hidden = np.matmul(one_hot, np_embedding)
print(hidden, np_embedding[token_idx])

# tf one hot matmul
hidden = tf.matmul(np.array([one_hot]), np.array(np_embedding))  # tf 특성상 2차원 이상을 입력 해야 함
print(hidden, np_embedding[token_idx])

# tf embedding
d_model = 4  # hidden length
tf_embedding = tf.keras.layers.Embedding(n_vocab, d_model)
print(tf_embedding.get_weights())  # 최초에는 초기화 되지 않음

# one hot lookup
token_idx = 5
hidden = tf_embedding(token_idx)
weights = tf_embedding.get_weights()  # 한번 실행 후 초기화 됨
print(weights)  # 최초에는 초기화 되지 않음
print(hidden, weights[0][token_idx])

# tf embedding by 초기 값
d_model = 4  # hidden length
tf_embedding = tf.keras.layers.Embedding(n_vocab, d_model, weights=np.array([np_embedding]))

# embedding lookup
token_idx = 5
hidden = tf_embedding(np.array([token_idx]))
weights = tf_embedding.get_weights()
print(weights)  # 최초에는 초기화 되지 않음
print(hidden, weights[0][token_idx])

# embedding by tf.gatehr (https://www.tensorflow.org/api_docs/python/tf/gather)
hidden = tf.gather(weights[0], np.array([token_idx]))
print(hidden, weights[0][token_idx])

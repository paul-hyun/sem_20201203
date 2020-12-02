# -*- coding:utf-8 -*-

import os

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#
# hidden
#

# hidden (bs, seq, d_model)
hidden = np.random.randint(1, 100, (2, 4, 8)) / 100
hidden = hidden.astype(np.float32)
print(hidden)

#
# basic usage (xW + b)
#

# output (bs, seq, n_output)
dense_1 = tf.keras.layers.Dense(4)
output_1 = dense_1(hidden)
print(output_1)

# dense weights
weights = dense_1.get_weights()
print(weights)
W = weights[0]
b = weights[1]

#
# matmul usage (xW + b)
#

# shape of hidden
bs, n_seq, d_model = tf.shape(hidden)
print(bs, n_seq, d_model)

# reshape
hidden_r = tf.reshape(hidden, [-1, d_model])
print(hidden)
print(hidden_r)

# xW + b
output_2_r = tf.matmul(hidden_r, W) + b
print(output_2_r)

# reshape
output_2 = tf.reshape(output_2_r, [bs, n_seq, -1])
print(output_2)
print(output_1)

# 결과 비교
print(np.array_equal(output_1, output_2))
print(output_1 - output_2)

#
# with init weights (xW + b)
#

# init weight
W = np.random.randint(1, 100, (8, 4)) / 100
W = W.astype(np.float32)
print(W)

b = np.random.randint(1, 100, (4,)) / 100
b = b.astype(np.float32)
print(b)

# dens with weights (xW + b)
dense_2 = tf.keras.layers.Dense(4, weights=[W, b])
output_3 = dense_2(hidden)
print(output_3)

#
# by mat mul (xW + b)
#

# shape check
bs, n_seq, d_model = tf.shape(hidden)
print(bs, n_seq, d_model)

# reshpae
hidden_r = tf.reshape(hidden, [-1, d_model])
print(hidden)
print(hidden_r)

# xW + b
output_4_r = tf.matmul(hidden_r, W) + b
print(output_4_r)

# reshape
output_4 = tf.reshape(output_4_r, [bs, n_seq, -1])
print(output_3)
print(output_4)

# 결과 값이 같음
print(np.array_equal(output_3, output_4))
print(output_3 - output_4)

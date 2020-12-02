# -*- coding:utf-8 -*-

import os

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#
# voector
#

vector = np.random.randint(1, 10, (2, 3)).astype(np.float32)
vector_1 = vector[0]
vector_2 = vector[1]
print(vector_1, vector_2)

# length
# https://www.tensorflow.org/api_docs/python/tf/norm
length_11 = tf.norm(vector_1)
print(length_11)
length_21 = tf.norm(vector_2)
print(length_21)

# sqrt(sum(square(x)))
length_12 = np.sqrt(np.sum(np.square(vector_1)))
print(length_12)

length_22 = np.sqrt(np.sum(np.square(vector_2)))
print(length_22)

#
# cosine sim by dot
#

# a.b/|a||b|
output_1 = tf.matmul([vector_1], [vector_2], transpose_b=True) / (length_11 * length_21)
print(output_1)

#
# cosine sim by normalize and dot
#

# normalize vector_1
normal_11 = vector_1 / length_11
print(normal_11)
print(tf.norm(normal_11))

# normalize vector_1
# https://www.tensorflow.org/api_docs/python/tf/math/l2_normalize
normal_12 = tf.math.l2_normalize(vector_1)
print(normal_12)
print(tf.norm(normal_12))

# normalize vector_2
normal_21 = vector_2 / length_21
print(normal_21)
print(tf.norm(normal_21))

# normalize vector_2
normal_22 = tf.math.l2_normalize(vector_2)
print(normal_22)
print(tf.norm(normal_22))

output_2 = tf.matmul([normal_11], [normal_21], transpose_b=True)
print(output_2)

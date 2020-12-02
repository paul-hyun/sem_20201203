# -*- coding:utf-8 -*-

import os

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#
# hidden
#

# hidden (bs, seq, d_model)
hidden = np.random.randint(-99, 100, (2, 4, 8)) / 10
hidden = hidden.astype(np.float32)
print(hidden)

#
# softmax (tf)
#

# softmax
output_1 = tf.nn.softmax(hidden, axis=-1)
print(output_1)  # bs, seq, d_model

# sum of prob
output_2 = tf.reduce_sum(output_1, axis=-1)
print(output_2)  # bs, seq

#
# numpy softmax
#

# exp(x)
output_3 = np.exp(hidden)
print(output_3)

# sum(exp(x))
output_4 = np.sum(output_3, axis=-1)
output_4 = np.expand_dims(output_4, axis=-1)
print(output_4)

# exp(x) / sum(exp(x))
output_5 = output_3 / output_4
print(output_5)
print(output_1)
print(output_1 - output_5)

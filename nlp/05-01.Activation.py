# -*- coding:utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#
# value
# https://reniew.github.io/12/
# https://gaussian37.github.io/dl-concept-relu6/
#
value = np.array([i / 100 for i in range(-999, 1000)]).astype(np.float32)
print(value)

#
# sigmoid
#
hidden_11 = tf.nn.sigmoid(value)
print(hidden_11)

# draw plot
plt.plot(value, hidden_11)
plt.title('sigmoid')
plt.show()

# 1 / (1 + exp(-x))
hidden_12 = 1 / (1 + np.exp(-value))
print(hidden_12)

# draw plot
plt.plot(value, hidden_12)
plt.title('sigmoid')
plt.show()

#
# Relu
#
hidden_21 = tf.nn.relu(value)
print(hidden_21)

# draw plot
plt.plot(value, hidden_21)
plt.title('reul')
plt.show()

# max(0, x)
hidden_22 = np.maximum(0, value)
print(hidden_22)

# draw plot
plt.plot(value, hidden_22)
plt.title('relu')
plt.show()

#
# Relu
#
hidden_31 = tf.nn.relu6(value)
print(hidden_31)

# draw plot
plt.plot(value, hidden_31)
plt.title('reul6')
plt.show()

# min(6, max(0, x))
hidden_32 = np.minimum(6, np.maximum(0, value))
print(hidden_32)

# draw plot
plt.plot(value, hidden_32)
plt.title('relu6')
plt.show()

#
# tanh
#
hidden_41 = tf.nn.tanh(value)
print(hidden_41)

# draw plot
plt.plot(value, hidden_41)
plt.title('tanh')
plt.show()

# (exp(x) - exp(-x)) / (exp(x) + exp(-x))
hidden_42 = (np.exp(value) - np.exp(-value)) / (np.exp(value) + np.exp(-value))
print(hidden_42)

# draw plot
plt.plot(value, hidden_42)
plt.title('tanh')
plt.show()


#
# gelu
#


def gelu(x):
    """
    gelu activation 함수
    :param x: 입력 값
    :return: gelu activation result
    """
    return 0.5 * x * (1 + K.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))


# 0.5x(1 + tanh(x * sqrt(2/pi)(x + 0.044715 + x**3)
hidden_51 = gelu(value)
print(hidden_51)

# draw plot
plt.plot(value, hidden_51)
plt.title('gelu')
plt.show()

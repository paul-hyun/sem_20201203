# -*- coding:utf-8 -*-

import os

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#
# hidden
#

# hidden (bs, seq, d_model)
hidden = np.random.randint(1, 100, (2, 9, 8)) / 100
hidden = hidden.astype(np.float32)
print(hidden)

# CNN (padding valid)
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
#
conv_1 = tf.keras.layers.Conv1D(filters=4, kernel_size=3, padding='valid')  # 기본 값
output_1 = conv_1(hidden)  # (bs, n_seq - kernel_size + 1, filters)
print(output_1)

# weights
weights = conv_1.get_weights()
W = weights[0]
b = weights[1]
print(W.shape)  # (kernel_size, input_hidden, filters)
print(b.shape)  # (filters,)

#
# CNN (padding same) kernel size에 따라서 양 옆에 자동으로 padding(all zero)를 추가해서 길이를 맞춤
#
conv_2 = tf.keras.layers.Conv1D(filters=4, kernel_size=3, padding='same')
output_2 = conv_2(hidden)  # (bs, n_seq, filters)
print(output_2)

# weights
weights = conv_2.get_weights()
W = weights[0]
b = weights[1]
print(W.shape)  # (kernel_size, input_hidden, filters)
print(b.shape)  # (filters,)

#
# CNN (padding causal) kernel size에 첫음에 자동으로 padding(all zero)를 추가해서 길이를 맞춤
#
conv_3 = tf.keras.layers.Conv1D(filters=4, kernel_size=3, padding='causal')
output_3 = conv_3(hidden)  # (bs, n_seq, filters)
print(output_3)

# weights
weights = conv_3.get_weights()
W = weights[0]
b = weights[1]
print(W.shape)  # (kernel_size, input_hidden, filters)
print(b.shape)  # (filters,)

#
# CNN dilation_rate=2
#
conv_4 = tf.keras.layers.Conv1D(filters=4, kernel_size=3, dilation_rate=1)
output_4 = conv_4(hidden)  # (bs, n_seq - window_size + 1, filters), window_size = kernel_size + (kernel_size - 1) * (dilation_rate - 1)
print(output_4)

# weights
weights = conv_4.get_weights()
W = weights[0]
b = weights[1]
print(W.shape)  # (kernel_size, input_hidden, filters)
print(b.shape)  # (filters,)


#
# Stack CNN
#
def build_model_cnn_stack(n_vocab, d_model):
    """
    stack cnn model build
    :param n_vocab: number of vocab
    :param d_model: hidden size
    :return model: model object
    :return embedding: embedding object (화면 출력 용)
    """
    input = tf.keras.layers.Input(shape=(1,))

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)  # (n_vocab x d_model)
    hidden = embedding(input)  # (bs, 1, d_model)

    hidden = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding='causal')(hidden)
    hidden = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding='causal')(hidden)
    hidden = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding='causal')(hidden)
    hidden = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding='causal')(hidden)
    hidden = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding='causal')(hidden)
    output = tf.keras.layers.Dense(n_vocab, activation=tf.nn.softmax)(hidden)  # (bs, 1, n_vocab)

    model = tf.keras.Model(inputs=input, outputs=output)
    return model


model = build_model_cnn_stack(10, 8)
print(model.summary())



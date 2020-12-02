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

#
# Simple RN
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN
#

# default (return_sequences=False, return_state=False)
rnn_11 = tf.keras.layers.SimpleRNN(units=4)
output_11 = rnn_11(hidden)  # (bs, units)
print(output_11)

weights = rnn_11.get_weights()
Wx = weights[0]
Wh = weights[1]
b = weights[2]
print(Wx.shape)  # (d_model, unit)
print(Wx)
print(Wh.shape)  # (unit, unit)
print(Wh)
print(b.shape)  # (unit, unit)
print(b)

# (return_sequences=True, return_state=False)
rnn_12 = tf.keras.layers.SimpleRNN(units=4, return_sequences=True)
output_12 = rnn_12(hidden)  # (bs, seq, units)
print(output_12)

# (return_sequences=False, return_state=True)
rnn_13 = tf.keras.layers.SimpleRNN(units=4, return_state=True)
output_13, fh_13 = rnn_13(hidden)  # (bs, units), (bs, unit)
print(output_13)
print(fh_13)

# (return_sequences=True, return_state=True)
rnn_14 = tf.keras.layers.SimpleRNN(units=4, return_sequences=True, return_state=True)
output_14, fh_14 = rnn_14(hidden)  # (bs, seq, units), (bs, unit)
print(output_14)
print(fh_14)

# call with initial statue
output_11 = rnn_11(hidden, initial_state=fh_13)  # (bs, units)
print(output_11)
output_12 = rnn_12(hidden, initial_state=fh_13)  # (bs, seq, units)
print(output_12)
output_13, _ = rnn_13(hidden, initial_state=fh_14)  # (bs, units)
print(output_13)
output_14, _ = rnn_14(hidden, initial_state=fh_14)  # (bs, seq, units)
print(output_14)

#
# LSTM
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
#

# default (return_sequences=False, return_state=False)
lstm_21 = tf.keras.layers.LSTM(units=4)
output_21 = lstm_21(hidden)  # (bs, units)
print(output_21)

weights = lstm_21.get_weights()
print(len(weights))
Wx = weights[0]
Wh = weights[1]
b = weights[2]
print(Wx.shape)  # (d_model, unit * 4) (Wxf, Wxi, Wxc, Wxo)
print(Wx)
print(Wh.shape)  # (unit, unit * 4) (Whf, Whi, Whc, Who)
print(Wh)
print(b.shape)  # (unit * 4) (bf, bi, bc, bo)
print(b)

# shape of hidden
bs, n_seq, d_model = tf.shape(hidden)
print(bs, n_seq, d_model)

# (seq, unit * 4)
xval_1 = tf.matmul(hidden, Wx)
print(xval_1.shape)
xval_2 = tf.reshape(xval_1, (bs, n_seq, 4, -1))
print(xval_2.shape)

# (return_sequences=True, return_state=False)
lstm_22 = tf.keras.layers.LSTM(units=4, return_sequences=True)
output_22 = lstm_22(hidden)  # (bs, seq, units)
print(output_22)

# (return_sequences=False, return_state=True)
lstm_23 = tf.keras.layers.LSTM(units=4, return_state=True)
output_23, fh_23, fc_23 = lstm_23(hidden)  # (bs, units), (bs, unit)
print(output_23)
print(fh_23)
print(fc_23)

# (return_sequences=True, return_state=True)
lstm_24 = tf.keras.layers.LSTM(units=4, return_sequences=True, return_state=True)
output_24, fh_24, fc_24 = lstm_24(hidden)  # (bs, seq, units), (bs, unit)
print(output_24)
print(fh_24)
print(fc_24)

# call with initial statue
output_21 = lstm_21(hidden, initial_state=(fh_23, fc_23))  # (bs, units)
print(output_21)
output_22 = lstm_22(hidden, initial_state=(fh_23, fc_23))  # (bs, seq, units)
print(output_22)
output_23, _, _ = lstm_23(hidden, initial_state=(fh_24, fc_24))  # (bs, units)
print(output_23)
output_24, _, _ = lstm_24(hidden, initial_state=(fh_24, fc_24))  # (bs, seq, units)
print(output_24)

#
# GRU
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU
#

# default (return_sequences=False, return_state=False)
gru_31 = tf.keras.layers.GRU(units=4)
output_31 = gru_31(hidden)  # (bs, units)
print(output_31)

weights = gru_31.get_weights()
print(len(weights))
Wx = weights[0]
Wh = weights[1]
b = weights[2]
print(Wx.shape)  # (d_model, unit * 3) (Wxr, Wxz, Wxg)
print(Wx)
print(Wh.shape)  # (unit, unit * 3) (Whr, Whz, Whg)
print(Wh)
print(b.shape)  # (2, unit * 3) (bxr, bxz, bxg),(bhr, bhz, bhg)
print(b)

# shape of hidden
bs, n_seq, d_model = tf.shape(hidden)
print(bs, n_seq, d_model)

# (seq, unit * 3)
xval_1 = tf.matmul(hidden, Wx)
print(xval_1.shape)
xval_2 = tf.reshape(xval_1, (bs, n_seq, 3, -1))
print(xval_2.shape)

# (return_sequences=True, return_state=False)
gru_32 = tf.keras.layers.GRU(units=4, return_sequences=True)
output_32 = gru_32(hidden)  # (bs, seq, units)
print(output_32)

# (return_sequences=False, return_state=True)
gru_33 = tf.keras.layers.GRU(units=4, return_state=True)
output_33, fh_33 = gru_33(hidden)  # (bs, units), (bs, unit)
print(output_33)
print(fh_33)

# (return_sequences=True, return_state=True)
gru_34 = tf.keras.layers.GRU(units=4, return_sequences=True, return_state=True)
output_34, fh_34 = gru_34(hidden)  # (bs, seq, units), (bs, unit)
print(output_34)
print(fh_34)

# call with initial statue
output_31 = gru_31(hidden, initial_state=(fh_33))  # (bs, units)
print(output_31)
output_32 = gru_32(hidden, initial_state=(fh_33))  # (bs, seq, units)
print(output_32)
output_33, _ = gru_33(hidden, initial_state=(fh_34))  # (bs, units)
print(output_33)
output_34, _ = gru_34(hidden, initial_state=(fh_34))  # (bs, seq, units)
print(output_34)

#
# BiRNN
#
# default (return_sequences=False, return_state=False)
rnn_41 = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=4))
output_41 = rnn_41(hidden)  # (bs, units * 2)
print(output_41)

weights = rnn_41.get_weights()
Wx = weights[0]
Wh = weights[1]
b = weights[2]
print(Wx.shape)  # (d_model, unit)
print(Wx)
print(Wh.shape)  # (unit, unit)
print(Wh)
print(b.shape)  # (unit, unit)
print(b)

# (return_sequences=True, return_state=False)
rnn_42 = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=4, return_sequences=True))
output_42 = rnn_42(hidden)  # (bs, seq, units * 2)
print(output_42)

# (return_sequences=False, return_state=True)
rnn_43 = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=4, return_state=True))
output_43, fh_43, bh_43 = rnn_43(hidden)  # (bs, units * 2), (bs, unit), (bs, unit)
print(output_43)
print(fh_43)
print(bh_43)

# (return_sequences=True, return_state=True)
rnn_44 = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=4, return_sequences=True, return_state=True))
output_44, fh_44, bh_44 = rnn_44(hidden)  # (bs, seq, units * 2), (bs, unit), (bs, unit)
print(output_44)
print(fh_44)
print(bh_44)

# call with initial statue
output_41 = rnn_41(hidden, initial_state=(fh_43, bh_43))  # (bs, units * 2)
print(output_41)
output_42 = rnn_42(hidden, initial_state=(fh_43, bh_43))  # (bs, seq, units * 2)
print(output_42)
output_43, _, _ = rnn_43(hidden, initial_state=(fh_44, bh_44))  # (bs, units * 2)
print(output_43)
output_44, _, _ = rnn_44(hidden, initial_state=(fh_44, bh_44))  # (bs, seq, units * 2)
print(output_44)

#
# BiLSTM
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
#

# default (return_sequences=False, return_state=False)
lstm_51 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=4))
output_51 = lstm_51(hidden)  # (bs, units * 2)
print(output_51)

weights = lstm_51.get_weights()
print(len(weights))
Wx = weights[0]
Wh = weights[1]
b = weights[2]
print(Wx.shape)  # (d_model, unit * 4) (Wxf, Wxi, Wxc, Wxo)
print(Wx)
print(Wh.shape)  # (unit, unit * 4) (Whf, Whi, Whc, Who)
print(Wh)
print(b.shape)  # (unit * 4) (bf, bi, bc, bo)
print(b)

# shape of hidden
bs, n_seq, d_model = tf.shape(hidden)
print(bs, n_seq, d_model)

# (seq, unit * 4)
xval_1 = tf.matmul(hidden, Wx)
print(xval_1.shape)
xval_2 = tf.reshape(xval_1, (bs, n_seq, 4, -1))
print(xval_2.shape)

# (return_sequences=True, return_state=False)
lstm_52 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=4, return_sequences=True))
output_52 = lstm_52(hidden)  # (bs, seq, units * 2)
print(output_52)

# (return_sequences=False, return_state=True)
lstm_53 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=4, return_state=True))
output_53, fh_53, fc_53, bh_53, bc_53 = lstm_53(hidden)  # (bs, units * 2), (bs, unit), (bs, unit), (bs, unit), (bs, unit)
print(output_53)
print(fh_53)
print(fc_53)
print(bh_53)
print(bc_53)

# (return_sequences=True, return_state=True)
lstm_54 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=4, return_sequences=True, return_state=True))
output_54, fh_54, fc_54, bh_54, bc_54 = lstm_54(hidden)  # (bs, seq, units * 2), (bs, unit), (bs, unit), (bs, unit), (bs, unit)
print(output_54)
print(fh_54)
print(fc_54)
print(bh_54)
print(bc_54)

# call with initial statue
output_51 = lstm_51(hidden, initial_state=(fh_53, fc_53, bh_53, bc_53))  # (bs, units * 2)
print(output_51)
output_52 = lstm_52(hidden, initial_state=(fh_53, fc_53, bh_53, bc_53))  # (bs, seq, units * 2)
print(output_52)
output_53, _, _, _, _ = lstm_53(hidden, initial_state=(fh_54, fc_54, bh_54, bc_54))  # (bs, units * 2)
print(output_53)
output_54, _, _, _, _ = lstm_54(hidden, initial_state=(fh_54, fc_54, bh_54, bc_54))  # (bs, seq, units * 2)
print(output_54)

#
# GRU
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU
#

# default (return_sequences=False, return_state=False)
gru_61 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=4))
output_61 = gru_61(hidden)  # (bs, units * 2)
print(output_61)

weights = gru_61.get_weights()
print(len(weights))
Wx = weights[0]
Wh = weights[1]
b = weights[2]
print(Wx.shape)  # (d_model, unit * 3) (Wxr, Wxz, Wxg)
print(Wx)
print(Wh.shape)  # (unit, unit * 3) (Whr, Whz, Whg)
print(Wh)
print(b.shape)  # (2, unit * 3) (bxr, bxz, bxg),(bhr, bhz, bhg)
print(b)

# shape of hidden
bs, n_seq, d_model = tf.shape(hidden)
print(bs, n_seq, d_model)

# (seq, unit * 3)
xval_1 = tf.matmul(hidden, Wx)
print(xval_1.shape)
xval_2 = tf.reshape(xval_1, (bs, n_seq, 3, -1))
print(xval_2.shape)

# (return_sequences=True, return_state=False)
gru_62 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=4, return_sequences=True))
output_62 = gru_62(hidden)  # (bs, seq, units * 2)
print(output_62)

# (return_sequences=False, return_state=True)
gru_63 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=4, return_state=True))
output_63, fh_63, bh_63 = gru_63(hidden)  # (bs, units), (bs, unit)
print(output_63)
print(fh_63)
print(bh_63)

# (return_sequences=True, return_state=True)
gru_64 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=4, return_sequences=True, return_state=True))
output_64, fh_64, bh_64 = gru_64(hidden)  # (bs, seq, units), (bs, unit)
print(output_64)
print(fh_64)
print(bh_64)

# call with initial statue
output_61 = gru_61(hidden, initial_state=(fh_63, bh_63))  # (bs, units)
print(output_61)
output_62 = gru_62(hidden, initial_state=(fh_63, bh_63))  # (bs, seq, units)
print(output_62)
output_63, _, _ = gru_63(hidden, initial_state=(fh_64, bh_64))  # (bs, units)
print(output_63)
output_64, _, _ = gru_64(hidden, initial_state=(fh_64, bh_64))  # (bs, seq, units)
print(output_64)

#
# Stack RNN
#
def build_model_rnn_stack(n_vocab, d_model):
    """
    stack rnn model build
    :param n_vocab: number of vocab
    :param d_model: hidden size
    :return model: model object
    :return embedding: embedding object (화면 출력 용)
    """
    input = tf.keras.layers.Input(shape=(1,))

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)  # (n_vocab x d_model)
    hidden = embedding(input)  # (bs, 1, d_model)

    hidden = tf.keras.layers.SimpleRNN(units=d_model, return_sequences=True)(hidden)
    hidden = tf.keras.layers.LSTM(units=d_model, return_sequences=True)(hidden)
    hidden = tf.keras.layers.GRU(units=d_model, return_sequences=True)(hidden)
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=d_model, return_sequences=True))(hidden)
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=d_model, return_sequences=True))(hidden)
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=d_model, return_sequences=True))(hidden)
    output = tf.keras.layers.Dense(n_vocab, activation=tf.nn.softmax)(hidden)  # (bs, 1, n_vocab)

    model = tf.keras.Model(inputs=input, outputs=output)
    return model


model = build_model_rnn_stack(10, 8)
print(model.summary())


#
# Stack CNN & RNN
#
def build_model_cnn_rnn_stack(n_vocab, d_model):
    """
    stack rnn model build
    :param n_vocab: number of vocab
    :param d_model: hidden size
    :return model: model object
    :return embedding: embedding object (화면 출력 용)
    """
    input = tf.keras.layers.Input(shape=(1,))

    embedding = tf.keras.layers.Embedding(n_vocab, d_model)  # (n_vocab x d_model)
    hidden = embedding(input)  # (bs, 1, d_model)

    hidden = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding='causal')(hidden)
    hidden = tf.keras.layers.SimpleRNN(units=d_model, return_sequences=True)(hidden)
    hidden = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding='causal')(hidden)
    hidden = tf.keras.layers.LSTM(units=d_model, return_sequences=True)(hidden)
    hidden = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding='causal')(hidden)
    hidden = tf.keras.layers.GRU(units=d_model, return_sequences=True)(hidden)
    hidden = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding='causal')(hidden)
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=d_model, return_sequences=True))(hidden)
    hidden = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding='causal')(hidden)
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=d_model, return_sequences=True))(hidden)
    hidden = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding='causal')(hidden)
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=d_model, return_sequences=True))(hidden)
    output = tf.keras.layers.Dense(n_vocab, activation=tf.nn.softmax)(hidden)  # (bs, 1, n_vocab)

    model = tf.keras.Model(inputs=input, outputs=output)
    return model


model = build_model_cnn_rnn_stack(10, 8)
print(model.summary())

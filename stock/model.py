# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras.backend as K


def build_model_rnn(n_seq, d_model, n_output=1):
    """
    RNN Model
    :param n_seq: number of vocab
    :param d_model: hidden size
    :param n_output: number of output
    :return model: model object
    """
    inputs = tf.keras.layers.Input((n_seq, d_model))  # (bs, n_seq, d_model)

    rnn = tf.keras.layers.SimpleRNN(units=50, activation=tf.nn.relu)
    hidden = rnn(inputs)  # (bs, n_seq, units)

    output_dense = tf.keras.layers.Dense(n_output)
    outputs = output_dense(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def build_model_rnn_code_embed(n_seq, d_model, n_code, d_code, n_output=1):
    """
    RNN Model whth Embedding
    :param n_seq: number of vocab
    :param d_model: hidden size
    :param n_code: number of code
    :param d_code: dim of code embed
    :param n_output: number of output
    :return model: model object
    """
    inputs_1 = tf.keras.layers.Input((n_seq, d_model))  # (bs, n_seq, d_model)
    inputs_2 = tf.keras.layers.Input((n_seq,))  # (bs, 1)

    code_embed = tf.keras.layers.Embedding(n_code, d_code)
    code_hidden = code_embed(inputs_2)  # (bs, n_seq, d_code)

    hidden = K.concatenate([inputs_1, code_hidden], axis=-1)  # (bs, n_seq, d_model + d_code)

    rnn = tf.keras.layers.SimpleRNN(units=50, activation=tf.nn.relu)  # (bs, n_seq, 50)
    hidden = rnn(hidden)  # (bs, units)

    output_dense = tf.keras.layers.Dense(n_output)
    outputs = output_dense(hidden)  # (bs, n_output)

    model = tf.keras.Model(inputs=(inputs_1, inputs_2), outputs=outputs)
    return model


def build_model_rnn_seq2seq(n_seq, d_model, n_code, d_code, y_step, n_output=1):
    """
    Seq2Seq Model
    :param n_seq: number of vocab
    :param d_model: hidden size
    :param n_code: number of code
    :param d_code: dim of code embed
    :param n_output: number of output
    :return model: model object
    """
    enc_inputs_1 = tf.keras.layers.Input((n_seq, d_model))  # (bs, n_seq, d_model)
    enc_inputs_2 = tf.keras.layers.Input((n_seq,))  # (bs, 1)
    dec_inputs = tf.keras.layers.Input((y_step, 1))

    code_embed = tf.keras.layers.Embedding(n_code, d_code)
    code_hidden = code_embed(enc_inputs_2)  # (bs, n_seq, d_code)

    hidden = K.concatenate([enc_inputs_1, code_hidden], axis=-1)  # (bs, n_seq, d_model + d_code)

    enc_rnn = tf.keras.layers.SimpleRNN(units=64, return_state=True, activation=tf.nn.relu)
    _, fw_h = enc_rnn(hidden)  # (bs, units)

    dec_rnn = tf.keras.layers.SimpleRNN(units=64, return_sequences=True, activation=tf.nn.relu)
    hidden = dec_rnn(dec_inputs, initial_state=[fw_h])  # (bs, y_step, units)

    output_dense = tf.keras.layers.Dense(n_output)
    outputs = output_dense(hidden)  # (bs, y_step, n_output)

    model = tf.keras.Model(inputs=(enc_inputs_1, enc_inputs_2, dec_inputs), outputs=outputs)
    return model


class DotProductAttention(tf.keras.layers.Layer):
    """
    dot product attention class
    """

    def __init__(self, **kwargs):
        """
        init class
        :param kwargs: args
        """
        super().__init__(**kwargs)

    def call(self, inputs):
        """
        run layer
        :param inputs: enc_input, dec_input tuple
        :return attn_out: attention output
        """
        Q, K, V = inputs
        # attention score (dot-product)
        attn_score = tf.matmul(Q, K, transpose_b=True)
        # attention prov
        attn_prob = tf.nn.softmax(attn_score, axis=-1)
        # weighted sum
        attn_out = tf.matmul(attn_prob, V)
        return attn_out


def build_model_rnn_seq2seq_dot(n_seq, d_model, n_code, d_code, y_step, n_output=1):
    """
    Seq2Seq Attention Model
    :param n_seq: number of vocab
    :param d_model: hidden size
    :param n_code: number of code
    :param d_code: dim of code embed
    :param n_output: number of output
    :return model: model object
    """
    enc_inputs_1 = tf.keras.layers.Input((n_seq, d_model))  # (bs, n_seq, d_model)
    enc_inputs_2 = tf.keras.layers.Input((n_seq,))  # (bs, 1)
    dec_inputs = tf.keras.layers.Input((y_step, 1))

    code_embed = tf.keras.layers.Embedding(n_code, d_code)
    code_hidden = code_embed(enc_inputs_2)  # (bs, n_seq, d_code)

    hidden = K.concatenate([enc_inputs_1, code_hidden], axis=-1)  # (bs, n_seq, d_model + d_code)

    enc_rnn = tf.keras.layers.SimpleRNN(units=64, return_state=True, return_sequences=True, activation=tf.nn.relu)
    enc_hidden, fw_h = enc_rnn(hidden)  # (bs, n_seq, units)

    dec_rnn = tf.keras.layers.SimpleRNN(units=64, return_sequences=True, activation=tf.nn.relu)
    dec_hidden = dec_rnn(dec_inputs, initial_state=[fw_h])  # (bs, y_step, units)

    attn = DotProductAttention()
    attn_out = attn((dec_hidden, enc_hidden, enc_hidden))  # bs, y_step, units
    hidden = tf.concat([dec_hidden, attn_out], axis=-1)  # bs, y_step, 2 * units

    output_dense = tf.keras.layers.Dense(n_output)
    outputs = output_dense(hidden)  # (bs, y_step, n_output)

    model = tf.keras.Model(inputs=(enc_inputs_1, enc_inputs_2, dec_inputs), outputs=outputs)
    return model


def build_model_lstm_gru_attn(n_seq, d_model, n_code, d_code, n_output=1):
    """
    RNN Model whth Embedding
    :param n_seq: number of vocab
    :param d_model: hidden size
    :param n_code: number of code
    :param d_code: dim of code embed
    :param n_output: number of output
    :return model: model object
    """
    inputs_1 = tf.keras.layers.Input((n_seq, d_model))  # (bs, n_seq, d_model)
    inputs_2 = tf.keras.layers.Input((n_seq,))  # (bs, 1)

    code_embed = tf.keras.layers.Embedding(n_code, d_code)
    code_hidden = code_embed(inputs_2)  # (bs, n_seq, d_code)

    hidden = K.concatenate([inputs_1, code_hidden], axis=-1)  # (bs, n_seq, d_model + d_code)

    lstm = tf.keras.layers.LSTM(units=64, return_sequences=True, activation=tf.nn.relu)  # (bs, n_seq, units)
    hidden_lstm = lstm(hidden)  # (bs, n_seq, units)
    gru = tf.keras.layers.GRU(units=64, return_sequences=True, activation=tf.nn.relu)  # (bs, n_seq, units)
    hidden_gru = gru(hidden_lstm)  # (bs, n_seq, units)

    attn = DotProductAttention()
    attn_lstm = attn((hidden_lstm, hidden_lstm, hidden_lstm))
    attn_gru = attn((hidden_gru, hidden_gru, hidden_gru))

    hidden_lstm = tf.keras.layers.GlobalMaxPool1D()(hidden_lstm)  # (bs, units)
    hidden_gru = tf.keras.layers.GlobalMaxPool1D()(hidden_gru)  # (bs, units)
    attn_lstm = tf.keras.layers.GlobalMaxPool1D()(attn_lstm)  # (bs, units)
    attn_gru = tf.keras.layers.GlobalMaxPool1D()(attn_gru)  # (bs, units)

    hidden = tf.concat([hidden_lstm, hidden_gru, attn_lstm, attn_gru], axis=-1)  # (bs, units * 4)
    hidden = tf.keras.layers.Dense(64, activation=tf.nn.relu)(hidden)

    output_dense = tf.keras.layers.Dense(n_output)
    outputs = output_dense(hidden)  # (bs, n_output)

    model = tf.keras.Model(inputs=(inputs_1, inputs_2), outputs=outputs)
    return model

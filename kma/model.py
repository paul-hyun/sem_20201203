import tensorflow as tf


def build_model_base_lstm(config):
    x_seq = config['data']['x_seq']
    x_cols = config['data']['x_cols']

    inputs = tf.keras.layers.Input((x_seq, len(x_cols)))  # bs, x_seq, d_model
    hidden = inputs

    hidden = tf.keras.layers.LSTM(units=config['model']['lstm1_unit'], return_sequences=True)(hidden)  # (bs, x_seq, units)
    hidden = tf.keras.layers.LSTM(units=config['model']['lstm2_unit'], return_sequences=True)(hidden)  # (bs, x_seq, units)
    hidden = tf.keras.layers.LSTM(units=config['model']['lstm3_unit'], return_sequences=True)(hidden)  # (bs, x_seq, units)
    hidden = tf.keras.layers.Dropout(config['model']['dropout'])(hidden)

    output_dense = tf.keras.layers.Dense(1)
    outputs = output_dense(hidden)

    model = tf.keras.Model(inputs=(inputs), outputs=outputs)
    return model


def build_model(config):
    name = config['model']['name']
    if name == 'base_lstm':
        return build_model_base_lstm(config)
    else:
        raise ValueError(f'unknown model name: {name}')

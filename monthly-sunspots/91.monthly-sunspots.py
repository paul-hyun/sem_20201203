# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data_dir = './data'
if not os.path.exists(data_dir):
    data_dir = '../data'
print(os.listdir(data_dir))

sunspots_dir = os.path.join(data_dir, 'sunspots')
os.listdir(sunspots_dir)

train_start = pd.to_datetime('1749-01-01')
train_end = pd.to_datetime('1979-12-01 ')
test_start = pd.to_datetime('1980-01-01 ')
test_end = pd.to_datetime('1983-12-01 ')
train_start, test_end

x_cols = ['LogSunspots']
y_col = 'LogSunspots'

df = pd.read_csv(os.path.join(sunspots_dir, 'monthly-sunspots.csv'))
df
df.info()

df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df
df.info()

df = df[(train_start <= df['Month']) & (df['Month'] <= test_end)]
df
df.info()
df.describe()

df_test = df[(test_start <= df['Month']) & (df['Month'] <= test_end)]
df_test

df = df.set_index('Month')
df
df.info()

df['Sunspots'].plot(figsize=(12, 5))
plt.show()

df_test = df_test.set_index('Month')
df_test
df_test.info()

df_test['Sunspots'].plot(figsize=(12, 5))
plt.show()

df['Y'] = df.index.year
df['M'] = df.index.month
df['Q'] = df.index.quarter

df.boxplot(column='Sunspots', by='Y', grid=True, figsize=(12, 5))
plt.show()

df.boxplot(column='Sunspots', by='M', grid=True, figsize=(12, 5))
plt.show()

df.boxplot(column='Sunspots', by='Q', grid=True, figsize=(12, 5))
plt.show()

df = df.reset_index()
df
df.info()

df['EpSunspots'] = df['Sunspots'] + 1
df
df.info()
df.describe()

df['LogSunspots'] = np.log(df['EpSunspots'])
df
df.info()
df.describe()


class MinMaxScaler():
    def __init__(self, min_val, max_val):
        assert (max_val > min_val)
        self.min_val = min_val
        self.max_val = max_val

    def scale_value(self, val):
        return (val - self.min_val) / (self.max_val - self.min_val)

    def inv_scale_value(self, scaled_val):
        return self.min_val + scaled_val * (self.max_val - self.min_val)


scaler_dic = {}
for col in x_cols:
    min_val = df[col].min()
    max_val = df[col].max()
    scaler_dic[col] = MinMaxScaler(min_val, max_val)

n_seq = 24
train_inputs, train_labels = [], []
test_inputs, test_labels = [], []
for i in range(0, len(df) - n_seq):
    x = []
    for j in range(n_seq):
        xj = df.iloc[i + j]
        xh = []
        for col in x_cols:
            x_scaler = scaler_dic[col]
            xh.append(x_scaler.scale_value(xj[col]))
        x.append(xh)
    x = df.iloc[i:i + n_seq][x_cols].to_numpy()
    y_val = df.iloc[i + n_seq][y_col]
    y_scaler = scaler_dic[y_col]
    y = y_scaler.scale_value(y_val)
    date = df.iloc[i + n_seq]['Month']
    if train_start <= date <= train_end:
        train_inputs.append(x)
        train_labels.append(y)
    elif test_start <= date <= test_end:
        test_inputs.append(x)
        test_labels.append(y)
    else:
        print(f'discard {date}')

train_inputs = np.array(train_inputs)
train_labels = np.array(train_labels)
test_inputs = np.array(test_inputs)
test_labels = np.array(test_labels)

train_inputs.shape, train_labels.shape, test_inputs.shape, test_labels.shape


def build_model_rnn(n_seq, d_model, n_output=1):
    """
    RNN Model
    :param n_seq: number of vocab
    :param d_model: hidden size
    :param n_output: number of output
    :return model: model object
    """
    inputs = tf.keras.layers.Input((n_seq, d_model))  # (bs, n_seq, d_model)

    hidden = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='causal')(inputs)
    hidden = tf.keras.layers.SimpleRNN(units=32, return_sequences=True, activation=tf.nn.relu)(hidden)  # (bs, n_seq, units)
    hidden = tf.keras.layers.SimpleRNN(units=64, return_sequences=True, activation=tf.nn.relu)(hidden)  # (bs, n_seq, units)
    hidden = tf.keras.layers.SimpleRNN(units=128, return_sequences=True, activation=tf.nn.relu)(hidden)  # (bs, n_seq, units)
    hidden = tf.keras.layers.Dropout(0.2)(hidden)
    hidden = tf.keras.layers.SimpleRNN(units=64, return_sequences=True, activation=tf.nn.relu)(hidden)  # (bs, n_seq, units)
    hidden = tf.keras.layers.SimpleRNN(units=32, activation=tf.nn.relu)(hidden)  # (bs, units)
    hidden = tf.keras.layers.Dropout(0.2)(hidden)

    output_dense = tf.keras.layers.Dense(n_output)
    outputs = output_dense(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


model = build_model_rnn(n_seq, len(x_cols))
model.summary()

model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())

# early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# save weights
save_weights = tf.keras.callbacks.ModelCheckpoint(os.path.join(sunspots_dir, 'rnn.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_freq='epoch', save_weights_only=True)
# train
history = model.fit(train_inputs, train_labels, epochs=100, batch_size=32, validation_data=(test_inputs, test_labels), callbacks=[early_stopping, save_weights])

model = build_model_rnn(n_seq, len(x_cols))
model.summary()
model.load_weights(save_weights.filepath)

#
# test eval
#
test_preds = model.predict(test_inputs)
test_preds.shape
test_preds = tf.squeeze(test_preds)
test_preds
test_labels

y_scaler = scaler_dic[y_col]
test_labels_log = np.array([y_scaler.inv_scale_value(v) for v in test_labels])
test_labels_log
test_preds_log = np.array([y_scaler.inv_scale_value(v) for v in test_preds])
test_preds_log
test_labels_org = np.exp(test_labels_log)
test_labels_org
test_preds_org = np.exp(test_preds_log)
test_preds_org

plt.figure(figsize=(16, 4))
plt.plot(test_labels_org, 'b-', label='y_true')
plt.plot(test_preds_org, 'r--', label='y_pred')
plt.legend()
plt.show()

plt.figure(figsize=(16, 4))
plt.plot(test_labels_org - test_preds_org, 'g-', label='y_diff')
plt.legend()
plt.show()

rmse = tf.sqrt(tf.keras.losses.MSE(test_labels_org, test_preds_org))
mae = tf.keras.losses.MAE(test_labels_org, test_preds_org)
mape = tf.keras.losses.MAPE(test_labels_org, test_preds_org)
print(pd.DataFrame([rmse, mae, mape], index=['RMSE', 'MAE', 'MAPE']).head())

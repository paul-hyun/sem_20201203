# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from data import read_marcap, load_datas
from model import build_model_rnn

data_dir = './data'
if not os.path.exists(data_dir):
    data_dir = '../data'
print(os.listdir(data_dir))

marcap_dir = os.path.join(data_dir, 'marcap')
marcap_data = os.path.join(marcap_dir, 'data')
os.listdir(marcap_data)

train_start = pd.to_datetime('2000-01-01')
train_end = pd.to_datetime('2020-06-30')
test_start = pd.to_datetime('2020-08-01')
test_end = pd.to_datetime('2020-10-31')
train_start, test_end

# 삼성전기 code '009150'
df_sem = read_marcap(train_start, test_end, ['009150'], marcap_data)
df_sem.drop(df_sem[df_sem['Marcap'] == 0].index, inplace=True)
df_sem['LogMarcap'] = np.log(df_sem['Marcap'])
df_sem

n_seq = 10

x_cols = ['LogMarcap']
y_col = 'LogMarcap'
train_inputs, train_labels, test_inputs, test_labels = load_datas(df_sem, x_cols, y_col, train_start, train_end, test_start, test_end, n_seq)
train_inputs.shape, train_labels.shape, test_inputs.shape, test_labels.shape

model = build_model_rnn(n_seq, len(x_cols))
model.summary()

model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())

# early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# save weights
save_weights = tf.keras.callbacks.ModelCheckpoint(os.path.join('1corp_logmarcap.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_freq='epoch', save_weights_only=True)
# save weights
csv_log = tf.keras.callbacks.CSVLogger(os.path.join('1corp_logmarcap.csv'), separator=',', append=False)
# train
history = model.fit(train_inputs, train_labels, epochs=100, batch_size=32, validation_data=(test_inputs, test_labels), callbacks=[early_stopping, save_weights, csv_log])


model = build_model_rnn(n_seq, len(x_cols))
model.summary()
model.load_weights(save_weights.filepath)

#
# train eval
#
n_batch = 32
train_preds = []
for i in range(0, len(train_inputs), n_batch):
    batch_inputs = train_inputs[i:i + n_batch]
    y_pred = model.predict(batch_inputs)
    y_pred = y_pred.squeeze(axis=-1)
    train_preds.extend(y_pred)
train_preds = np.array(train_preds)
assert len(train_labels) == len(train_preds)
train_labels.shape, train_preds.shape

train_labels_log = train_labels
train_preds_log = train_preds

plt.figure(figsize=(16, 4))
plt.plot(train_labels_log, 'b-', label='y_true')
plt.plot(train_preds_log, 'r--', label='y_pred')
plt.legend()
plt.show()

plt.figure(figsize=(16, 4))
plt.plot(train_labels_log - train_preds_log, 'g-', label='y_diff')
plt.legend()
plt.show()

# https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-17-%ED%9A%8C%EA%B7%80-%ED%8F%89%EA%B0%80-%EC%A7%80%ED%91%9C
# https://m.blog.naver.com/PostView.nhn?blogId=limitsinx&logNo=221578145366&proxyReferer=https:%2F%2Fwww.google.com%2F
rmse = tf.sqrt(tf.keras.losses.MSE(train_labels_log, train_preds_log))
mae = tf.keras.losses.MAE(train_labels_log, train_preds_log)
mape = tf.keras.losses.MAPE(train_labels_log, train_preds_log)
print(pd.DataFrame([rmse, mae, mape], index=['RMSE', 'MAE', 'MAPE']).head())

#
# test eval
#
test_preds = []
for i in range(0, len(test_inputs), n_batch):
    batch_inputs = test_inputs[i:i + n_batch]
    y_pred = model.predict(batch_inputs)
    y_pred = y_pred.squeeze(axis=-1)
    test_preds.extend(y_pred)
test_preds = np.array(test_preds)
assert len(test_labels) == len(test_preds)
test_labels.shape, test_preds.shape

test_labels_log = test_labels
test_preds_log = test_preds

plt.figure(figsize=(16, 4))
plt.plot(test_labels_log, 'b-', label='y_true')
plt.plot(test_preds_log, 'r--', label='y_pred')
plt.legend()
plt.show()

plt.figure(figsize=(16, 4))
plt.plot(test_labels_log - test_preds_log, 'g-', label='y_diff')
plt.legend()
plt.show()

# https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-17-%ED%9A%8C%EA%B7%80-%ED%8F%89%EA%B0%80-%EC%A7%80%ED%91%9C
# https://m.blog.naver.com/PostView.nhn?blogId=limitsinx&logNo=221578145366&proxyReferer=https:%2F%2Fwww.google.com%2F
rmse = tf.sqrt(tf.keras.losses.MSE(test_labels_log, test_preds_log))
mae = tf.keras.losses.MAE(test_labels_log, test_preds_log)
mape = tf.keras.losses.MAPE(test_labels_log, test_preds_log)
print(pd.DataFrame([rmse, mae, mape], index=['RMSE', 'MAE', 'MAPE']).head())

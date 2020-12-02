# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import tensorflow as tf

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
df_sem

n_seq = 10

x_cols = ['Marcap']
y_col = 'Marcap'
train_inputs, train_labels, test_inputs, test_labels = load_datas(df_sem, x_cols, y_col, train_start, train_end, test_start, test_end, n_seq)
train_inputs.shape, train_labels.shape, test_inputs.shape, test_labels.shape

model = build_model_rnn(n_seq, len(x_cols))
model.summary()

model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())

# early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# save weights
save_weights = tf.keras.callbacks.ModelCheckpoint(os.path.join('1corp_marcap.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_freq='epoch', save_weights_only=True)
# save weights
csv_log = tf.keras.callbacks.CSVLogger(os.path.join('1corp_marcap.csv'), separator=',', append=False)
# train
history = model.fit(train_inputs, train_labels, epochs=100, batch_size=32, validation_data=(test_inputs, test_labels), callbacks=[early_stopping, save_weights, csv_log])

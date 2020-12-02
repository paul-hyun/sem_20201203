# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from tqdm import tqdm, trange


# https://mkjjo.github.io/python/2019/01/10/scaler.html
class MinMaxScaler():
    def __init__(self, min_val, max_val):
        assert (max_val > min_val)
        self.min_val = min_val
        self.max_val = max_val

    def scale_value(self, val):
        return (val - self.min_val) / (self.max_val - self.min_val)

    def inv_scale_value(self, scaled_val):
        return self.min_val + scaled_val * (self.max_val - self.min_val)


def read_marcap(start, end, codes, marcap_data):
    dfs = []
    for year in range(start.year, end.year + 1):
        csv_file = os.path.join(marcap_data, f'marcap-{year}.csv.gz')
        df = pd.read_csv(csv_file, dtype={'Code': str})
        dfs.append(df)
    # 데이터 합치기
    df_all = pd.concat(dfs)
    # string을 date로 변환
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    # codes 적용
    df_all = df_all[df_all['Code'].isin(codes)]
    # date 기간 적용
    df_all = df_all[(start <= df_all["Date"]) & (df_all["Date"] <= end)]
    # date 순으로 정렬
    df_all = df_all.sort_values('Date', ascending=True)
    return df_all


def load_datas(df, x_cols, y_col, train_start, train_end, test_start, test_end, n_seq):
    train_inputs, train_labels = [], []
    test_inputs, test_labels = [], []
    for i in trange(0, len(df) - n_seq):
        x = df.iloc[i:i + n_seq][x_cols].to_numpy()
        y = df.iloc[i + n_seq][y_col]
        date = df.iloc[i + n_seq]['Date']
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

    return train_inputs, train_labels, test_inputs, test_labels


def load_datas_scaled(df, x_cols, y_col, train_start, train_end, test_start, test_end, n_seq):
    scaler_dic = {}
    for col in x_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        scaler_dic[col] = MinMaxScaler(min_val, max_val)

    train_inputs, train_labels = [], []
    test_inputs, test_labels = [], []
    for i in trange(0, len(df) - n_seq):
        x = []
        for j in range(n_seq):
            xj = df.iloc[i + j]
            xh = []
            for col in x_cols:
                x_scaler = scaler_dic[col]
                xh.append(x_scaler.scale_value(xj[col]))
            x.append(xh)
        y_scaler = scaler_dic[y_col]
        y = y_scaler.scale_value(df.iloc[i + n_seq][y_col])
        date = df.iloc[i + n_seq]['Date']
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

    return train_inputs, train_labels, test_inputs, test_labels, scaler_dic


def _load_datas_by_code_x_multi(df, code, x_cols, y_col, n_seq, scaler_dic):
    df_code = df[df['Code'] == code]

    data_dic = {}
    for i in trange(0, len(df_code) - n_seq):
        x = []
        for j in range(n_seq):
            xj = df_code.iloc[i + j]
            xh = []
            for col in x_cols:
                x_scaler = scaler_dic[col]
                xh.append(x_scaler.scale_value(xj[col]))
            x.append(xh)
        y_scaler = scaler_dic[y_col]
        y = y_scaler.scale_value(df_code.iloc[i + n_seq][y_col])
        date = df_code.iloc[i + n_seq]['Date']

        data_dic[date] = (x, y)

    return data_dic


def load_datas_scaled_x_multi(df, code_to_id, y_code, x_cols, y_col, train_start, train_end, test_start, test_end, n_seq):
    scaler_dic = {}
    for col in x_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        scaler_dic[col] = MinMaxScaler(min_val, max_val)

    train_inputs, train_codes, train_labels = [], [], []
    test_inputs, test_codes, test_labels = [], [], []

    data_code_dic = {}
    for code in code_to_id.keys():
        data_dic = _load_datas_by_code_x_multi(df, code, x_cols, y_col, n_seq, scaler_dic)
        data_code_dic[code] = data_dic

    date_list = df['Date'].unique()
    for i, date in enumerate(tqdm(date_list)):
        date = pd.to_datetime(date)
        for code in code_to_id.keys():
            data_dic = data_code_dic[code]
            if date in data_dic:
                x, y = data_dic[date]

                if train_start <= date <= train_end:
                    train_inputs.append(x)
                    train_codes.append([code_to_id[code]] * n_seq)
                    train_labels.append(y)
                elif test_start <= date <= test_end and code == y_code:
                    test_inputs.append(x)
                    test_codes.append([code_to_id[code]] * n_seq)
                    test_labels.append(y)
                else:
                    print(f'discard {date} / {code}')
            else:
                print(f'not exists {date} / {code}')

    train_inputs = np.array(train_inputs)
    train_codes = np.array(train_codes)
    train_labels = np.array(train_labels)
    test_inputs = np.array(test_inputs)
    test_codes = np.array(test_codes)
    test_labels = np.array(test_labels)

    return train_inputs, train_codes, train_labels, test_inputs, test_codes, test_labels, scaler_dic


def _load_datas_by_code_x_y_multi(df, code, x_cols, y_cols, n_seq, scaler_dic):
    df_code = df[df['Code'] == code]

    data_dic = {}
    for i in trange(0, len(df_code) - n_seq):
        x = []
        for j in range(n_seq):
            xj = df_code.iloc[i + j]
            xh = []
            for col in x_cols:
                x_scaler = scaler_dic[col]
                xh.append(x_scaler.scale_value(xj[col]))
            x.append(xh)
        y = []
        yj = df_code.iloc[i + n_seq]
        for col in y_cols:
            y_scaler = scaler_dic[col]
            y.append(y_scaler.scale_value(yj[col]))
        date = df_code.iloc[i + n_seq]['Date']

        data_dic[date] = (x, y)

    return data_dic


def load_datas_scaled_x_y_multi(df, code_to_id, y_code, x_cols, y_cols, train_start, train_end, test_start, test_end, n_seq):
    scaler_dic = {}
    for col in x_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        scaler_dic[col] = MinMaxScaler(min_val, max_val)

    train_inputs, train_codes, train_labels = [], [], []
    test_inputs, test_codes, test_labels = [], [], []

    data_code_dic = {}
    for code in code_to_id.keys():
        data_dic = _load_datas_by_code_x_y_multi(df, code, x_cols, y_cols, n_seq, scaler_dic)
        data_code_dic[code] = data_dic

    date_list = df['Date'].unique()
    for i, date in enumerate(tqdm(date_list)):
        date = pd.to_datetime(date)
        for code in code_to_id.keys():
            data_dic = data_code_dic[code]
            if date in data_dic:
                x, y = data_dic[date]

                if train_start <= date <= train_end:
                    train_inputs.append(x)
                    train_codes.append([code_to_id[code]] * n_seq)
                    train_labels.append(y)
                elif test_start <= date <= test_end and code == y_code:
                    test_inputs.append(x)
                    test_codes.append([code_to_id[code]] * n_seq)
                    test_labels.append(y)
                else:
                    print(f'discard {date} / {code}')
            else:
                print(f'not exists {date} / {code}')

    train_inputs = np.array(train_inputs)
    train_codes = np.array(train_codes)
    train_labels = np.array(train_labels)
    test_inputs = np.array(test_inputs)
    test_codes = np.array(test_codes)
    test_labels = np.array(test_labels)

    return train_inputs, train_codes, train_labels, test_inputs, test_codes, test_labels, scaler_dic


def _load_datas_by_code_x_y_step(df, code, x_cols, y_col, n_seq, y_step, scaler_dic):
    df_code = df[df['Code'] == code]

    data_dic = {}
    for i in trange(0, len(df_code) - n_seq - y_step + 1):
        x = []
        for j in range(n_seq):
            xj = df_code.iloc[i + j]
            xh = []
            for col in x_cols:
                x_scaler = scaler_dic[col]
                xh.append(x_scaler.scale_value(xj[col]))
            x.append(xh)
        y = []
        for j in range(y_step):
            yj = df_code.iloc[i + n_seq + j]
            y_scaler = scaler_dic[y_col]
            y.append(y_scaler.scale_value(yj[y_col]))
        date = df_code.iloc[i + n_seq]['Date']

        data_dic[date] = (x, y)

    return data_dic


def load_datas_scaled_x_y_step(df, code_to_id, y_code, x_cols, y_col, train_start, train_end, test_start, test_end, n_seq, y_step):
    scaler_dic = {}
    for col in x_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        scaler_dic[col] = MinMaxScaler(min_val, max_val)

    train_inputs, train_codes, train_labels = [], [], []
    test_inputs, test_codes, test_labels = [], [], []

    data_code_dic = {}
    for code in code_to_id.keys():
        data_dic = _load_datas_by_code_x_y_step(df, code, x_cols, y_col, n_seq, y_step, scaler_dic)
        data_code_dic[code] = data_dic

    date_list = df['Date'].unique()
    for i, date in enumerate(tqdm(date_list)):
        date = pd.to_datetime(date)
        for code in code_to_id.keys():
            data_dic = data_code_dic[code]
            if date in data_dic:
                x, y = data_dic[date]

                if train_start <= date <= train_end:
                    train_inputs.append(x)
                    train_codes.append([code_to_id[code]] * n_seq)
                    train_labels.append(y)
                elif test_start <= date <= test_end and code == y_code:
                    test_inputs.append(x)
                    test_codes.append([code_to_id[code]] * n_seq)
                    test_labels.append(y)
                else:
                    print(f'discard {date} / {code}')
            else:
                print(f'not exists {date} / {code}')

    train_inputs = np.array(train_inputs)
    train_codes = np.array(train_codes)
    train_labels = np.array(train_labels)
    test_inputs = np.array(test_inputs)
    test_codes = np.array(test_codes)
    test_labels = np.array(test_labels)

    return train_inputs, train_codes, train_labels, test_inputs, test_codes, test_labels, scaler_dic


def _load_datas_by_code_x_y_s2s(df, code, x_cols, y_col, n_seq, y_step, scaler_dic):
    df_code = df[df['Code'] == code]

    data_dic = {}
    for i in trange(0, len(df_code) - n_seq - y_step + 1):
        x = []
        for j in range(n_seq):
            xj = df_code.iloc[i + j]
            xh = []
            for col in x_cols:
                x_scaler = scaler_dic[col]
                xh.append(x_scaler.scale_value(xj[col]))
            x.append(xh)
        y = []
        for j in range(y_step + 1):
            yj = df_code.iloc[i + n_seq - 1 + j]
            y_scaler = scaler_dic[y_col]
            y.append([y_scaler.scale_value(yj[y_col])])
        date = df_code.iloc[i + n_seq]['Date']

        data_dic[date] = (x, y[:-1], y[1:])

    return data_dic


def load_datas_scaled_x_y_s2s(df, code_to_id, y_code, x_cols, y_col, train_start, train_end, test_start, test_end, n_seq, y_step):
    scaler_dic = {}
    for col in x_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        scaler_dic[col] = MinMaxScaler(min_val, max_val)

    train_enc_inputs, train_codes, train_dec_inputs, train_labels = [], [], [], []
    test_enc_inputs, test_codes, test_dec_inputs, test_labels = [], [], [], []

    data_code_dic = {}
    for code in code_to_id.keys():
        data_dic = _load_datas_by_code_x_y_s2s(df, code, x_cols, y_col, n_seq, y_step, scaler_dic)
        data_code_dic[code] = data_dic

    date_list = df['Date'].unique()
    for i, date in enumerate(tqdm(date_list)):
        date = pd.to_datetime(date)
        for code in code_to_id.keys():
            data_dic = data_code_dic[code]
            if date in data_dic:
                enc_x, dec_x, y = data_dic[date]

                if train_start <= date <= train_end:
                    train_enc_inputs.append(enc_x)
                    train_codes.append([code_to_id[code]] * n_seq)
                    train_dec_inputs.append(dec_x)
                    train_labels.append(y)
                elif test_start <= date <= test_end and code == y_code:
                    test_enc_inputs.append(enc_x)
                    test_codes.append([code_to_id[code]] * n_seq)
                    test_dec_inputs.append(dec_x)
                    test_labels.append(y)
                else:
                    print(f'discard {date} / {code}')
            else:
                print(f'not exists {date} / {code}')

    train_enc_inputs = np.array(train_enc_inputs)
    train_codes = np.array(train_codes)
    train_dec_inputs = np.array(train_dec_inputs)
    train_labels = np.array(train_labels)
    test_enc_inputs = np.array(test_enc_inputs)
    test_codes = np.array(test_codes)
    test_dec_inputs = np.array(test_dec_inputs)
    test_labels = np.array(test_labels)

    return train_enc_inputs, train_codes, train_dec_inputs, train_labels, test_enc_inputs, test_codes, test_dec_inputs, test_labels, scaler_dic


import pandas as pd
import numpy as np


def read_dataframe(config, data_file):
    """
    데이터 조회
    :param config: config
    :param data_file: data file
    :return: 기간 내의 dataframe
    """
    # data frame 읽어오기
    df = pd.read_csv(data_file, encoding='CP949')
    # 일시를 날짜 형식으로 변경
    df['일시'] = pd.to_datetime(df['일시'], format='%Y-%m')
    # 지점 필터
    if config['data']['x_locations']:
        df = df[df['지점'].isin(config['data']['x_locations'])]
    # 기간 필터
    start = pd.to_datetime(config['data']['train_start'])
    end = pd.to_datetime(config['data']['test_end'])
    df = df[(start <= df['일시']) & (df['일시'] <= end)]
    # 날짜 순으로 정렬
    df = df.sort_values('일시', ascending=True)
    return df


def _load_col_matrix(df, x_col):
    df_pivot = df.pivot_table(index=['지점'],
                              values=x_col,
                              columns=['일시'],
                              aggfunc='sum')
    df_pivot = df_pivot.fillna(0)
    df_pivot = df_pivot.reset_index()

    df_matrix = df_pivot.drop(['지점'], axis=1).values
    df_matrix = np.expand_dims(df_matrix, axis=-1)
    return df_matrix


def load_data(df, config):
    x_seq = config['data']['x_seq']
    x_cols = config['data']['x_cols']

    test_start = pd.to_datetime(config['data']['test_start'])
    test_end = pd.to_datetime(config['data']['test_end'])
    df_test = df[(test_start <= df["일시"]) & (df["일시"] <= test_end)]
    test_len = len(df_test['일시'].unique())

    df_matrix_list = []
    for x_col in x_cols:
        df_tmp = _load_col_matrix(df, x_col)
        df_matrix_list.append(df_tmp)
    if 1 < len(df_matrix_list):
        df_matrix = np.concatenate(df_matrix_list, axis=-1)
    else:
        df_matrix = df_matrix_list[0]

    bs, n_total, d_model = df_matrix.shape

    total_inputs, total_labels = [], []
    for i in range(n_total - x_seq):
        x = df_matrix[:, i:i + x_seq]
        y = df_matrix[:, i + 1:i + x_seq + 1, :1]
        total_inputs.append(x)
        total_labels.append(y)

    train_inputs = total_inputs[:-test_len]
    train_inputs = np.concatenate(train_inputs, axis=0)
    train_labels = total_labels[:-test_len]
    train_labels = np.concatenate(train_labels, axis=0)

    test_inputs = total_inputs[-test_len:]
    test_inputs = np.concatenate(test_inputs, axis=0)
    test_labels = total_labels[-test_len:]
    test_labels = np.concatenate(test_labels, axis=0)

    return (train_inputs, train_labels), (test_inputs, test_labels)


# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""# Data
- https://github.com/FinanceData/marcap

- Date : 날짜 (DatetimeIndex)
- Code : 종목코드
- Name : 종명이름
- Open : 시가
- High : 고가
- Low : 저가
- Close : 종가
- Volume : 거래량
- Amount : 거래대금
- Changes : 전일대비
- ChagesRatio : 전일비
- Marcap : 시가총액(백만원)
- Stocks : 상장주식수
- MarcapRatio : 시가총액비중(%)
- ForeignShares : 외국인 보유주식수
- ForeignRatio : 외국인 지분율(%)
- Rank: 시가총액 순위 (당일)
"""

marcap_data = './marcap/data'
os.listdir(marcap_data)

train_start = pd.to_datetime('2000-01-01')
train_end = pd.to_datetime('2018-12-31')
test_start = pd.to_datetime('2019-01-01')
test_end = pd.to_datetime('2020-11-15')
train_start, test_end

# 연도별 데이터 조회
dfs = []
for year in range(train_start.year, test_end.year + 1):
    csv_file = os.path.join(marcap_data, f'marcap-{year}.csv.gz')
    df = pd.read_csv(csv_file, dtype={'Code': str})
    dfs.append(df)
dfs[-1]

# 데이터 합치기
df_all = pd.concat(dfs)
df_all

df_all.info()

# string을 date로 변환
df_all['Date'] = pd.to_datetime(df_all['Date'])  # index 용
df_all.info()

# date 기간 적용
df_all = df_all[(train_start <= df_all["Date"]) & (df_all["Date"] <= test_end)]
df_all.info()

# index 적용
df_all.set_index('Date', inplace=True)
df_all.info()

# 시간순 정렬
df_all.sort_index(ascending=True, inplace=True)
df_all

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
# 25%, 50%, 75% 는 백분위 지수 (percentile)
df_all.describe().T
# df_all.describe(include='all').T

# null check
df_all.isnull().sum()

# null rows
df_all[df_all.isnull().sum(axis=1) > 0]

df_all.index.value_counts()

# df_all[['Code', 'Name']].value_counts()

"""# 삼성전기 주가 이해하기"""

df_sem = df_all[df_all['Name'] == '삼성전기']
df_sem

# 시가총액(백만원) / 시가총액비중(%) / 상장주식수 / 시가총액 순위 (당일)
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(20, 20))
df_sem['Marcap'].plot(ax=axes[0])
df_sem['MarcapRatio'].plot(ax=axes[1])
df_sem['Stocks'].plot(ax=axes[2])
df_sem['Rank'].plot(ax=axes[3])
plt.show()

df_sem.drop(df_sem[df_sem['Marcap'] == 0].index, inplace=True)
df_sem['LogMarcap'] = np.log(df_sem['Marcap'])
# 시가총액(백만원) / 시가총액(log)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
df_sem['Marcap'].plot(ax=axes[0])
df_sem['LogMarcap'].plot(ax=axes[1])
plt.show()

# 시가 / 종가 / 고가 / 저가
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(20, 20))
df_sem['Open'].plot(ax=axes[0])
df_sem['Close'].plot(ax=axes[1])
df_sem['High'].plot(ax=axes[2])
df_sem['Low'].plot(ax=axes[3])
plt.show()

# 전일대비 / 전일비
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
df_sem['Changes'].plot(ax=axes[0])
df_sem['ChagesRatio'].plot(ax=axes[1])
plt.show()

df_sem.drop(df_sem[df_sem['Amount'] == 0].index, inplace=True)
df_sem['LogAmount'] = np.log(df_sem['Amount'])
# 거래량 / 거래금액
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 10))
df_sem['Volume'].plot(ax=axes[0])
df_sem['Amount'].plot(ax=axes[1])
df_sem['LogAmount'].plot(ax=axes[2])
plt.show()

# histogram
df_sem.hist(bins=25, grid=True, figsize=(16, 12))
plt.show()

df_sem.corr().style.background_gradient().set_precision(2)

plt.figure(figsize=(15, 15))
sns.heatmap(data=df_sem.corr(), annot=True, fmt='.2f', linewidths=.5, cmap='Blues')
plt.show()

df_sem['Year'] = df_sem.index.year
df_sem['Month'] = df_sem.index.month
df_sem['Day'] = df_sem.index.day
df_sem['Quater'] = df_sem.index.quarter

df_sem

df_sem.boxplot(column='LogMarcap', by='Year', grid=True, figsize=(12, 5))
plt.show()

df_sem.plot.scatter(y='LogMarcap', x='Year', grid=True, figsize=(12, 5))

df_sem.plot.scatter(y='LogMarcap', x='Year', c='Rank', grid=True, figsize=(12, 5), colormap='viridis')
plt.show()

df_sem.boxplot(column='LogMarcap', by='Month', grid=True, figsize=(12, 5))
plt.show()

df_sem.plot.scatter(y='LogMarcap', x='Month', grid=True, figsize=(12, 5))

df_sem.plot.scatter(y='LogMarcap', x='Month', c='Rank', grid=True, figsize=(12, 5), colormap='viridis')
plt.show()

df_sem.boxplot(column='LogMarcap', by='Day', grid=True, figsize=(12, 5))
plt.show()

df_sem.plot.scatter(y='LogMarcap', x='Day', grid=True, figsize=(12, 5))

df_sem.plot.scatter(y='LogMarcap', x='Day', c='Rank', grid=True, figsize=(12, 5), colormap='viridis')
plt.show()

df_sem.boxplot(column='LogMarcap', by='Quater', grid=True, figsize=(12, 5))
plt.show()

df_sem.plot.scatter(y='LogMarcap', x='Quater', grid=True, figsize=(12, 5))

df_sem.plot.scatter(y='LogMarcap', x='Quater', c='Rank', grid=True, figsize=(12, 5), colormap='viridis')
plt.show()

df_train = df_sem[(train_start <= df_sem.index) & (df_sem.index <= train_end)]
df_test = df_sem[(test_start <= df_sem.index) & (df_sem.index <= test_end)]
len(df_train), len(df_test)

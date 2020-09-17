# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 11. 11.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ksif.core.columns import *
from ksif.core.frame import QUANTILE
from scipy.stats.mstats import winsorize

FILE_NAME = '100-NN3_3-all-all-linear-he_uniform-glorot_uniform-none'
DATA_NAME = 'all'

predicted_ret_1 = 'predict_return_1'
chunk_num = 10
labels = [str(x) for x in range(1, chunk_num + 1)]

fml_result = pd.read_csv('prediction/{}.csv'.format(FILE_NAME))
fml_result = fml_result.drop(columns=[RET_1])
data = pd.read_csv('data/{}.csv'.format(DATA_NAME))

merged = fml_result.merge(data, on=[DATE, CODE])

merged[QUANTILE] = merged.groupby(by=[DATE])[predicted_ret_1].transform(
    lambda x: pd.qcut(x, chunk_num, labels=labels)
)
merged[QUANTILE] = merged[QUANTILE].apply(int).apply(str)

top = merged.loc[merged[QUANTILE] == '1', :]
others = merged.loc[merged[QUANTILE] != '1', :]

rolling_columns = [E_P, B_P, S_P, C_P, OP_P, GP_P, ROA, ROE, QROA, QROE, GP_A, ROIC, GP_S, SALESQOQ, GPQOQ, ROAQOQ,
                   MOM6, MOM12, BETA_1D, VOL_5M, LIQ_RATIO, EQUITY_RATIO, DEBT_RATIO, FOREIGN_OWNERSHIP_RATIO,
                   SHORT_SALE_VOLUME_RATIO]
rolled_columns = []
for column in rolling_columns:
    t_0 = column + '_t'
    t_1 = column + '_t-1'
    t_2 = column + '_t-2'
    t_3 = column + '_t-3'
    t_4 = column + '_t-4'
    t_5 = column + '_t-5'
    t_6 = column + '_t-6'
    t_7 = column + '_t-7'
    t_8 = column + '_t-8'
    t_9 = column + '_t-9'
    t_10 = column + '_t-10'
    t_11 = column + '_t-11'
    t_12 = column + '_t-12'
    rolled_columns.extend([t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10, t_11, t_12])

for factor in rolled_columns:
    winsorized_top = winsorize(top[factor], limits=[0.05, 0.05])
    winsorized_others = winsorize(others[factor], limits=[0.05, 0.05])
    min_lim = min(winsorized_top.min(), winsorized_others.min())
    max_lim = max(winsorized_top.max(), winsorized_others.max())
    bins = np.linspace(min_lim, max_lim, 100)
    plt.hist(top[factor], bins, alpha=0.5, label='top', weights=[9] * len(top[factor]))
    plt.axvline(x=top.loc[(top[factor] >= min_lim) & (top[factor] <= max_lim), factor].mean(), color='lightblue')
    plt.hist(others[factor], bins, alpha=0.5, label='others')
    plt.axvline(x=others.loc[(others[factor] >= min_lim) & (others[factor] <= max_lim), factor].mean(), color='orange')
    plt.legend(loc='upper right')
    plt.title('Top 10% vs. Others')
    plt.xlabel(factor)
    plt.savefig('histogram/{}.png'.format(factor))
    plt.show()

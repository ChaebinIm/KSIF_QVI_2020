# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2019-01-07
"""
import numpy as np
import pandas as pd
from ksif.core.columns import *

RET = 'return'
YYYYMM = 'year_month'


if __name__ == '__main__':
    method = 'Intersection'
    model_name = 'DNN8_1-all-linear-he_uniform-glorot_uniform-none-0.5'

    monthly_portfolio = pd.read_csv('data/portfolio.csv', parse_dates=[DATE])
    monthly_portfolio[DATE] += np.timedelta64(1, 'D')
    monthly_portfolio[YYYYMM] = monthly_portfolio[DATE].dt.year.astype(str) + \
                                monthly_portfolio[DATE].dt.month.astype(str).str.zfill(2)
    monthly_portfolio = monthly_portfolio[[CODE, YYYYMM]]

    daily_adjp = pd.read_csv('data/daily_adjp.csv', parse_dates=[DATE])
    daily_adjp.set_index([DATE], inplace=True)

    daily_returns = daily_adjp.pct_change(fill_method='ffill')
    daily_returns.reset_index(drop=False, inplace=True)
    daily_returns = pd.melt(daily_returns, id_vars=[DATE], var_name=CODE, value_name=RET)
    daily_returns.dropna(inplace=True)
    daily_returns[YYYYMM] = daily_returns[DATE].dt.year.astype(str) + \
                            daily_returns[DATE].dt.month.astype(str).str.zfill(2)

    daily_portfolio = pd.merge(monthly_portfolio, daily_returns, on=[CODE, YYYYMM])[[DATE, RET]]
    daily_returns = daily_portfolio.groupby(DATE).mean()
    daily_returns.to_csv('daily_backtest/{}-{}.csv'.format(method, model_name))

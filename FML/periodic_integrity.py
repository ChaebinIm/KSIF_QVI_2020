# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-12-20
"""
import pandas as pd
from ksif import Portfolio
from ksif.core.columns import *

if __name__ == '__main__':
    pf = Portfolio()

    total_periodic_integrity = pd.DataFrame()
    for factor in pf.columns[3:]:
        integrative_counts = pf.loc[:, [DATE, factor]].groupby(DATE).count()
        all_counts = pf.loc[:, [DATE, CODE]].groupby(DATE).count()
        all_counts.rename(columns={CODE: factor}, inplace=True)
        factor_periodic_integrity = integrative_counts / all_counts
        factor_periodic_integrity.fillna(0, inplace=True)
        total_periodic_integrity = pd.concat([total_periodic_integrity, factor_periodic_integrity], axis=1, sort=True)

    total_periodic_integrity.to_csv('periodic_integrity/periodic_integrity.csv', encoding='utf-8')

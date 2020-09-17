# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 11. 19.
"""
from data_generator import *

all_count = pd.read_csv('data/all.csv').groupby(by=[DATE])[CODE].count()
filter_count = pd.read_csv('data/filter.csv').groupby(by=[DATE])[CODE].count()
bollinger_count = pd.read_csv('data/bollinger.csv').groupby(by=[DATE])[CODE].count()
sector_count = pd.read_csv('data/sector.csv').groupby(by=[DATE])[CODE].count()


result = pd.DataFrame(data={
    ALL: all_count,
    FILTER: filter_count,
    BOLLINGER: bollinger_count,
    SECTOR: sector_count,
}).fillna(0)

if __name__ == '__main__':
    result.to_csv('universe_number/universe_number.csv')

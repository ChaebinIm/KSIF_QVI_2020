# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 6. 22.
"""
import datetime

import numpy as np
from pandas import Series


def last_day_of_month(any_day):
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
    return next_month - datetime.timedelta(days=next_month.day)


def zero_to_nan(series: Series) -> Series:
    return series.apply(lambda x: np.where(x == 0, np.nan, x))

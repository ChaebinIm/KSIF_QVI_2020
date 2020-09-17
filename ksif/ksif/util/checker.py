# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 9. 14.
"""
from functools import wraps
from pandas import DataFrame, Series

from ..errors import EmptyResultException


def not_empty(func):
    """
    Check the result of this function is empty or not.
    If the result is empty, raise a runtime error.
    """

    @wraps(func)
    def check_not_empty(*args, **kwargs):
        result = func(*args, **kwargs)
        if (type(result) is DataFrame or type(result) is Series) and result.empty:
            raise EmptyResultException()

        return result

    return check_not_empty

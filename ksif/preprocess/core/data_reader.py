# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 6. 19.
"""
import numpy as np
import pandas as pd
from pandas.io.excel import ExcelFile

from preprocess.core.columns import *

KIND = 'Kind'
SYMBOL = 'Symbol'
NAME = 'name'
ITEM_NAME = 'Item Name '
ITEM = 'Item'
FREQUENCY = 'Frequency'

SYMBOL_NAME = 'Symbol Name'

DIR = 'data/{}.xlsx'
COMPANY_UNNECESSARY_COLUMNS = [KIND, NAME, ITEM_NAME, ITEM, FREQUENCY]
BENCHMARK_UNNECESSARY_COLUMNS = [SYMBOL, KIND, ITEM, ITEM_NAME, FREQUENCY]


# noinspection PyShadowingNames
def read_companies(excel_file: ExcelFile) -> pd.DataFrame:
    """
    :param excel_file: (ExcelFile)

    :return melted_companies: (DataFrame)
         code   | (String)
         date   | (Datetime)
         name   | (String)
         ...
    """
    # Read excel file.
    raw_companies = excel_file.parse(COMPANY, skiprows=8)

    # Rename Symbol -> code, Symbol Name -> name
    raw_companies = raw_companies.rename(columns={
        'Symbol': CODE,
        'Symbol Name': NAME,
    })

    # Save symbol names and item names.
    names = raw_companies.drop_duplicates(subset=CODE, keep='last').loc[:, [CODE, NAME]]
    names = names.set_index(CODE)
    item_name_num = len(raw_companies.loc[:1000, ITEM_NAME].unique())
    item_names = raw_companies.loc[:item_name_num - 1, ITEM_NAME]

    # Remove unnecessary columns, for example, Symbol, Kind, Item, Item Name, Frequency
    raw_companies = raw_companies.drop(columns=COMPANY_UNNECESSARY_COLUMNS)

    # Melt every items.
    melted_companies = pd.DataFrame(columns=[CODE, DATE])
    melted_companies = melted_companies.set_index([CODE, DATE])
    for index, item_name in enumerate(item_names):
        # Melt raw_benchmark. Symbole name -> code, column names -> date
        item_companies = pd.melt(raw_companies.iloc[index::item_name_num, :], id_vars=[CODE], var_name=DATE,
                                 value_name=item_name)
        item_companies[DATE] = pd.to_datetime(item_companies[DATE], format='%Y-$m-%D')
        item_companies = item_companies.set_index([CODE, DATE])
        melted_companies = melted_companies.join(item_companies, how='outer')

    melted_companies = melted_companies.rename(columns=COMPANY_RENAMES)

    # Add the names of company.
    melted_companies = melted_companies.join(names)

    melted_companies = melted_companies.reset_index()
    melted_companies = melted_companies.sort_values([CODE, DATE])

    # IS_MANAGED, IS_SUSPENDED: '정지' -> True, na -> False
    melted_companies[IS_MANAGED] = melted_companies[IS_MANAGED].replace('관리', True)
    melted_companies[IS_MANAGED] = melted_companies[IS_MANAGED].fillna(False)
    melted_companies[IS_SUSPENDED] = melted_companies[IS_SUSPENDED].replace('정지', True)
    melted_companies[IS_SUSPENDED] = melted_companies[IS_SUSPENDED].fillna(False)

    # nan -> 0
    to_zero_columns = [
        CFO, ALLOWANCE_AR_, TRADING_VOLUME, RES_EXP, AR, DIVP, AP,
        NET_PERSONAL_PURCHASE, NET_NATIONAL_PURCHASE, NET_FINANCIAL_INVESTMENT_PURCHASE,
        NET_INSTITUTIONAL_FOREIGN_PURCHASE, NET_INSTITUTIONAL_PURCHASE, NET_ETC_FINANCE_PURCHASE,
        NET_ETC_CORPORATION_PURCHASE, NET_ETC_FOREIGN_PURCHASE, NET_REGISTERED_FOREIGN_PURCHASE,
        NET_INSURANCE_PURCHASE, NET_PRIVATE_FUND_PURCHASE, NET_PENSION_PURCHASE, NET_FOREIGN_PURCHASE,
        NET_BANK_PURCHASE, NET_TRUST_PURCHASE, SHORT_SALE_BALANCE, FOREIGN_OWNERSHIP_RATIO
    ]
    melted_companies.loc[:, to_zero_columns] = melted_companies.replace(np.nan, 0.0).loc[:, to_zero_columns]

    # There are no SHORT_SALE_BALANCE before 2016-06-30
    melted_companies.loc[melted_companies[DATE] < '2016-06-30', SHORT_SALE_BALANCE] = np.nan

    # Sort by code and date
    melted_companies = melted_companies.sort_values([CODE, DATE]).reset_index(drop=True)

    return melted_companies


# noinspection PyShadowingNames
def read_benchmarks(excel_file: ExcelFile) -> pd.DataFrame:
    """
    :param excel_file: (ExcelFile)

    :return melted_benchmarks: (DataFrame)
        code        | (String)
        date        | (Datetime)
        price_index | (float)
    """
    # Read excel file.
    raw_benchmarks = excel_file.parse(BENCHMARK, skiprows=8)
    raw_macro_from_monthly = excel_file.parse(MACRO_MONTHLY, skiprows=8)

    # Use only CD91
    raw_risk_free = raw_macro_from_monthly.loc[raw_macro_from_monthly[ITEM_NAME] == '시장금리:CD유통수익률(91)(%)', :]

    # Remove unnecessary columns, for example, Symbol, Kind, Item, Item Name, Frequency
    raw_benchmarks = raw_benchmarks.drop(columns=BENCHMARK_UNNECESSARY_COLUMNS)
    raw_risk_free = raw_risk_free.drop(columns=BENCHMARK_UNNECESSARY_COLUMNS)
    raw_risk_free[SYMBOL_NAME] = CD91

    # Melt benchmarks. Symbole name -> code, column names -> date
    melted_benchmarks = _melt(raw_benchmarks, PRICE_INDEX)
    melted_risk_free = _melt(raw_risk_free, PRICE_INDEX)

    # Calculate a risk free rate index
    melted_risk_free[PRICE_INDEX] = (((melted_risk_free[PRICE_INDEX] / 100) + 1) ** (1 / 12)).cumprod()
    melted_benchmarks = pd.concat([melted_benchmarks, melted_risk_free])

    # Sort by code and date
    melted_benchmarks = melted_benchmarks.sort_values([CODE, DATE]).reset_index(drop=True)

    return melted_benchmarks


# noinspection PyShadowingNames
def read_factors(excel_file: ExcelFile) -> pd.DataFrame:
    """
    :param excel_file: (ExcelFile)

    :return transposed_factors: (DataFrame)
        key
            date            | (Datetime)
        column
            BIG_HIGH        | (float) Size & Book Value(2X3) 대형 - High
            BIG_MEDIUM      | (float) Size & Book Value(2X3) 대형 - Medium
            BIG_LOW         | (float) Size & Book Value(2X3) 대형 - Low
            SMALL_HIGH      | (float) Size & Book Value(2X3) 소형 - High
            SMALL_MEDIUM    | (float) Size & Book Value(2X3) 소형 - Medium
            SMALL_LOW       | (float) Size & Book Value(2X3) 소형 - Low
    """
    # Read excel file.
    raw_factors = excel_file.parse(FACTOR, skiprows=8)

    # Remove unnecessary columns, for example, Symbol, Kind, Item, Item Name, Frequency
    raw_factors = raw_factors.drop(columns=BENCHMARK_UNNECESSARY_COLUMNS)

    raw_factors = raw_factors.set_index(SYMBOL_NAME)

    transposed_factors = raw_factors.T
    transposed_factors.index.name = DATE

    # % to float
    transposed_factors = transposed_factors / 100

    return transposed_factors


def _melt(raw, value_name):
    melted = pd.melt(raw, id_vars=[SYMBOL_NAME], var_name=DATE, value_name=value_name)
    melted[DATE] = pd.to_datetime(melted[DATE], format='%Y-$m-%D')
    melted = melted.rename(columns={SYMBOL_NAME: CODE})
    melted = melted.dropna()
    return melted


def read_macro_daily(excel_file: ExcelFile):
    # Read excel file.
    raw_macro_from_daily = excel_file.parse(MACRO_DAILY, skiprows=8)

    return raw_macro_from_daily


def read_macro_monthly(excel_file: ExcelFile):
    # Read excel file.
    raw_macro_from_monthly = excel_file.parse(MACRO_MONTHLY, skiprows=8)

    return raw_macro_from_monthly

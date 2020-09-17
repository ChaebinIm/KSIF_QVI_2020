# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-21
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from ksif import Portfolio
from ksif.core.columns import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tqdm import tqdm

# DATA_SET
ALL = 'all'
MACRO = 'macro'
FILTER = 'filter'
BOLLINGER = 'bollinger'
SECTOR = 'sector'

TRADING_CAPITAL = 'trading_capital'

START_DATE = '2004-05-31'
USED_PAST_MONTHS = 12  # At a time, use past 12 months data and current month data.


def get_data_set(portfolio, rolling_columns, dummy_columns=None, return_y=True, apply_scaling_return=False,
                 apply_scaling_rolling_columns=False):
    if return_y:
        result_columns = [DATE, CODE, RET_1]
    else:
        result_columns = [DATE, CODE]

    if dummy_columns:
        data_set = portfolio.sort_values(by=[CODE, DATE]).reset_index(drop=True)[
            result_columns + rolling_columns + dummy_columns]
    else:
        data_set = portfolio.sort_values(by=[CODE, DATE]).reset_index(drop=True)[result_columns + rolling_columns]

    # MinMaxScale_Return
    if apply_scaling_return:
        data_set[RET_1] = data_set.groupby(by=[DATE])[RET_1].apply(
            lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))

    if apply_scaling_rolling_columns:
        for i in rolling_columns:
            data_set[i] = data_set.groupby(by=[DATE])[i].apply(
                lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))

    for column in tqdm(rolling_columns):
        for i in range(0, USED_PAST_MONTHS + 1):
            column_i = column + '_t-{}'.format(i)
            result_columns.append(column_i)
            if i == 0:
                data_set[column_i] = data_set[column]
            else:
                data_set[column_i] = data_set.groupby(by=[CODE]).apply(lambda x: x[column].shift(i)).reset_index(drop=True)
            print(column_i)

    if dummy_columns:
        result_columns.extend(dummy_columns)

    data_set = data_set[result_columns]
    data_set = data_set.dropna().reset_index(drop=True)

    return data_set


def save_data(old_data: bool, portfolio: Portfolio, data_name: str, rolling_columns: list,
              dummy_columns: list = None,
              filtering_dataframe=None):
    print("Start saving {}...".format(data_name))

    portfolio = portfolio.sort_values(by=[CODE, DATE]).reset_index(drop=True)

    if old_data:
        # old data
        # RET_1이 존재하지 않는 마지막 달 제거
        old_portfolio = portfolio.loc[~pd.isna(portfolio['ret_1m']), :]
        old_set = get_data_set(old_portfolio, rolling_columns, dummy_columns)

        if isinstance(filtering_dataframe, pd.DataFrame) and not filtering_dataframe.empty:
            filtering_dataframe = filtering_dataframe[[DATE, CODE]]
            old_set = pd.merge(old_set, filtering_dataframe, on=[DATE, CODE])

        old_set.reset_index(drop=True).to_dataframe().to_hdf(
            'data/{}.h5'.format(data_name), key='df', format='table', mode='w'
        )
    else:
        # recent data
        recent_set = get_data_set(portfolio, rolling_columns, dummy_columns, return_y=False)
        # 마지막 달만 사용
        last_month = np.sort(recent_set[DATE].unique())[-1]
        recent_set = recent_set.loc[recent_set[DATE] == last_month, :]

        if isinstance(filtering_dataframe, pd.DataFrame) and not filtering_dataframe.empty:
            filtering_dataframe = filtering_dataframe[[DATE, CODE]]
            recent_set = pd.merge(recent_set, filtering_dataframe, on=[DATE, CODE])

        recent_set.reset_index(drop=True).to_dataframe().to_hdf(
            'data/{}_recent.h5'.format(data_name), key='df', format='table', mode='w'
        )


def save_all(only_old_data: bool):
    rolling_columns = [E_P, B_P, S_P, C_P, OP_P, GP_P, ROA, ROE, QROA, QROE, GP_A, ROIC, GP_S, SALESQOQ, GPQOQ, ROAQOQ,
                       MOM6, MOM12, BETA_1D, VOL_5M, LIQ_RATIO, EQUITY_RATIO, DEBT_RATIO, FOREIGN_OWNERSHIP_RATIO]
    portfolio = Portfolio()
    # 최소 시가총액 100억
    portfolio = portfolio.loc[portfolio['size'] > 10000000000, :]

    save_data(only_old_data, portfolio, ALL, rolling_columns)


def save_filter(only_old_data: bool):
    rolling_columns = [E_P, B_P, S_P, C_P, OP_P, GP_P, ROA, ROE, QROA, QROE, GP_A, ROIC, GP_S, SALESQOQ, GPQOQ, ROAQOQ,
                       MOM6, MOM12, BETA_1D, VOL_5M, LIQ_RATIO, EQUITY_RATIO, DEBT_RATIO, FOREIGN_OWNERSHIP_RATIO]
    portfolio = Portfolio()
    # 최소 시가총액 100억
    portfolio = portfolio.loc[portfolio[MKTCAP] > 10000000000, :]

    # 2 < PER < 10.0 (http://pluspower.tistory.com/9)
    portfolio = portfolio.loc[(portfolio[PER] < 10) & (portfolio[PER] > 2)]
    # 0.2 < PBR < 1.0
    portfolio = portfolio.loc[(portfolio[PBR] < 1) & (portfolio[PBR] > 0.2)]
    # 2 < PCR < 8
    portfolio = portfolio.loc[(portfolio[PCR] < 8) & (portfolio[PCR] > 2)]
    # 0 < PSR < 0.8
    portfolio = portfolio.loc[portfolio[PSR] < 0.8]

    save_data(only_old_data, portfolio, FILTER, rolling_columns)


def save_bollinger(only_old_data: bool):
    rolling_columns = [E_P, B_P, S_P, C_P, OP_P, GP_P, ROA, ROE, QROA, QROE, GP_A, ROIC, GP_S, SALESQOQ, GPQOQ, ROAQOQ,
                       MOM6, MOM12, BETA_1D, VOL_5M, LIQ_RATIO, EQUITY_RATIO, DEBT_RATIO, FOREIGN_OWNERSHIP_RATIO]
    portfolio = Portfolio()
    # 최소 시가총액 100억
    portfolio = portfolio.loc[portfolio[MKTCAP] > 10000000000, :]

    # Bollinger
    portfolio = portfolio.sort_values(by=[CODE, DATE]).reset_index(drop=True)
    portfolio['mean'] = portfolio.groupby(CODE)[ENDP].rolling(20).mean().reset_index(drop=True)
    portfolio['std'] = portfolio.groupby(CODE)[ENDP].rolling(20).std().reset_index(drop=True)
    portfolio[BOLLINGER] = portfolio['mean'] - 2 * portfolio['std']
    bollingers = portfolio.loc[portfolio[ENDP] < portfolio[BOLLINGER], [DATE, CODE]]

    save_data(only_old_data, portfolio, BOLLINGER, rolling_columns, filtering_dataframe=bollingers)


def save_sector(only_old_data: bool):
    columns = [DATE, CODE, RET_1]
    rolling_columns = [E_P, B_P, S_P, C_P, OP_P, GP_P, ROA, ROE, QROA, QROE, GP_A, ROIC, GP_S, SALESQOQ, GPQOQ, ROAQOQ,
                       MOM6, MOM12, BETA_1D, VOL_5M, LIQ_RATIO, EQUITY_RATIO, DEBT_RATIO, FOREIGN_OWNERSHIP_RATIO]
    columns.extend(rolling_columns)
    portfolio = Portfolio()
    # 최소 시가총액 100억
    portfolio = portfolio.loc[portfolio[MKTCAP] > 10000000000, :]

    # KRX_SECTOR가 존재하지 않는 데이터 제거
    portfolio.dropna(subset=[KRX_SECTOR], inplace=True)
    portfolio = portfolio.sort_values(by=[CODE, DATE]).reset_index(drop=True)

    # sector를 숫자로 나타냄
    label_encoder = LabelEncoder()
    labeled_sector = label_encoder.fit_transform(portfolio[KRX_SECTOR])
    krx_sectors = label_encoder.classes_
    # 숫자로 나타낸 것을 모스부호로 표현
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoded_sector = one_hot_encoder.fit_transform(labeled_sector.reshape(len(labeled_sector), 1))
    # 기존 데이터에 붙히기
    df_one_hot_encoded_sector = pd.DataFrame(one_hot_encoded_sector, columns=krx_sectors).reset_index(drop=True)
    portfolio[krx_sectors] = df_one_hot_encoded_sector
    krx_sectors = list(krx_sectors)
    save_data(only_old_data, portfolio, SECTOR, rolling_columns, krx_sectors)


def save_macro(only_old_data: bool):
    rolling_columns = [
        E_P, B_P, S_P, C_P, OP_P, GP_P, ROA, ROE, QROA, QROE, GP_A, ROIC, GP_S, SALESQOQ, GPQOQ, ROAQOQ,
        MOM6, MOM12, BETA_1D, VOL_5M, LIQ_RATIO, EQUITY_RATIO, DEBT_RATIO, FOREIGN_OWNERSHIP_RATIO,
        TERM_SPREAD_KOR, TERM_SPREAD_US, CREDIT_SPREAD_KOR, LOG_USD2KRW, LOG_CHY2KRW, LOG_EURO2KRW,
        TED_SPREAD, LOG_NYSE, LOG_NASDAQ, LOG_OIL
    ]
    portfolio = Portfolio()
    # 최소 시가총액 100억
    portfolio = portfolio.loc[portfolio[MKTCAP] > 10000000000, :]

    save_data(only_old_data, portfolio, MACRO, rolling_columns)


def save_concepts(old_data: bool):
    log_mktcap = 'log_mktcap'
    portfolio = Portfolio()
    portfolio[log_mktcap] = np.log(portfolio[MKTCAP])

    value_factors = [E_P, B_P, S_P, C_P, DIVP]
    size_factors = [log_mktcap]
    momentum_factors = [MOM1, MOM12]
    quality_factors = [ROA, ROE, ROIC, S_A, DEBT_RATIO, EQUITY_RATIO, LIQ_RATIO]
    volatility_factors = [VOL_1D]

    factor_groups = {}

    for value_factor, value_name in zip([value_factors, []], ['value_', '']):
        for size_factor, size_name in zip([size_factors, []], ['size_', '']):
            for momentum_factor, momentum_name in zip([momentum_factors, []], ['momentum_', '']):
                for quality_factor, quality_name in zip([quality_factors, []], ['quality_', '']):
                    for volatility_factor, volatility_name in zip([volatility_factors, []], ['volatility_', '']):
                        factor_group = []
                        factor_group.extend(value_factor)
                        factor_group.extend(size_factor)
                        factor_group.extend(momentum_factor)
                        factor_group.extend(quality_factor)
                        factor_group.extend(volatility_factor)
                        factor_names = []
                        factor_names.extend(value_name)
                        factor_names.extend(size_name)
                        factor_names.extend(momentum_name)
                        factor_names.extend(quality_name)
                        factor_names.extend(volatility_name)
                        factor_name = ''.join(factor_names)
                        if factor_name:
                            factor_groups[factor_name[:-1]] = factor_group

    factor_group_len = len(factor_groups)

    with Pool(os.cpu_count()) as p:
        rs = [p.apply_async(save_data, [old_data, pf, key, value]) for pf, (key, value) in zip(
            [portfolio for _ in range(factor_group_len)],
            factor_groups.items()
        )]
        for r in rs:
            r.wait()
        p.close()
        p.join()


if __name__ == '__main__':
    old_data = True  # true : back testing, false : rebalancing
    # save_concepts(old_data=old_data)
    save_all(old_data)
    # with Pool(os.cpu_count()) as p:
    #     results = [p.apply_async(func, [old_data]) for func in [
    #         save_all,
    #         save_macro,
    #         save_filter,
    #         save_bollinger,
    #         save_sector
    #     ]]
    #     for result in results:
    #         result.wait()
    #     p.close()
    #     p.join()
    # save_sector(old_data)

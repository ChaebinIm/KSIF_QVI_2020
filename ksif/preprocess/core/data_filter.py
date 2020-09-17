# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 6. 21.
"""
import math

import pandas as pd
from dateutil.relativedelta import relativedelta

from ksif.io.google_drive import query_google_spreadsheet, HOLDING_COMPANY_GID, DELISTED_COMPANY_GID, \
    MERGED_COMPANY_GID, RELISTED_COMPANY_GID
from preprocess.core.columns import *
from .utils import last_day_of_month


def filter_companies(unfiltered_companies: pd.DataFrame) -> pd.DataFrame:
    """
    Filter spac companies and delisted rows.
    Mark on holding companies.
    Add a 0 price row at the end of delisted companies.
    Add a same price row at the end of merged companies.

    :param unfiltered_companies: (DataFrame) A dataframe containing all kind of companies and rows.

    :return filtered_companies: (DataFrame)
    """
    # Select rows which OUTCST > 0.
    unfiltered_companies = unfiltered_companies.loc[unfiltered_companies[OUTCST] > 0, :].reset_index(drop=True)

    # Select rows which ENDP is not nan.
    unfiltered_companies = unfiltered_companies.loc[~pd.isnull(unfiltered_companies[ENDP]), :].reset_index(drop=True)

    # Remove delisted rows.
    unfiltered_companies = unfiltered_companies.loc[[not math.isnan(price) for price in unfiltered_companies[ENDP]]]
    unfiltered_companies = unfiltered_companies.loc[
        [not math.isnan(price) for price in unfiltered_companies[ADJP]]]

    # Remove Spac companies.
    unfiltered_companies = unfiltered_companies.loc[[SPAC not in name for name in unfiltered_companies[NAME]], :]

    # Make all date into the last day of month.
    unfiltered_companies[DATE] = unfiltered_companies[DATE].apply(last_day_of_month)

    # Read the list of holding companies.
    holdings = query_google_spreadsheet(HOLDING_COMPANY_GID)

    unfiltered_companies[HOLDING] = [code in list(holdings[CODE]) for code in unfiltered_companies[CODE]]

    # Save the last day for check new relisted companies.
    pre_last_day = sorted(unfiltered_companies[DATE].unique())[-1]

    # Set the next price to 0 when a company is delisted due to bankruptcy.
    delisted_companies = query_google_spreadsheet(DELISTED_COMPANY_GID)
    delisted_records = unfiltered_companies.loc[
        [code in delisted_companies[CODE].unique() for code in unfiltered_companies[CODE]]
    ]
    delisted_records = delisted_records.groupby(CODE).last().reset_index()
    delisted_records[DATE] = delisted_records[DATE].apply(lambda x: x + relativedelta(months=1))
    delisted_records[[ENDP, ADJP]] = 0
    unfiltered_companies = pd.concat([unfiltered_companies, delisted_records], axis=0, ignore_index=True)

    # Set the next price to the current price when a company is delisted due to M&A.
    merged_companies = query_google_spreadsheet(MERGED_COMPANY_GID)
    merged_records = unfiltered_companies.loc[
        [code in merged_companies[CODE].unique() for code in unfiltered_companies[CODE]]
    ]
    merged_records = merged_records.groupby(CODE).last().reset_index()
    merged_records[DATE] = merged_records[DATE].apply(lambda x: x + relativedelta(months=1))
    unfiltered_companies = pd.concat([unfiltered_companies, merged_records], axis=0, ignore_index=True)

    # Set is_suspended of a delisted period to True when a company had been delisted and was listed again.
    relisted_companies = query_google_spreadsheet(RELISTED_COMPANY_GID)
    for index, (code, name, start_date, end_date) in relisted_companies.iterrows():
        unfiltered_companies.loc[(unfiltered_companies[CODE] == code) &
                                 (unfiltered_companies[DATE] >= start_date) &
                                 (unfiltered_companies[DATE] < end_date), IS_SUSPENDED] = True

    # Remove 한미은행(A016830), 에스와이코퍼레이션(A008080) because of data corruption.
    unfiltered_companies = unfiltered_companies.loc[unfiltered_companies[CODE] != 'A016830', :]
    unfiltered_companies = unfiltered_companies.loc[unfiltered_companies[CODE] != 'A008080', :]

    # Check new relisted companies.
    post_last_day = sorted(unfiltered_companies[DATE].unique())[-1]
    if pre_last_day != post_last_day:
        raise ValueError("There are new relisted companies.\n{}".format(
            unfiltered_companies.loc[unfiltered_companies[DATE] == post_last_day, :]
        ))

    filtered_companies = unfiltered_companies.sort_values([CODE, DATE]).reset_index(drop=True)
    return filtered_companies

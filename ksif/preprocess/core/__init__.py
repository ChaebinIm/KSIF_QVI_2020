# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 6. 19.
"""
import argparse

import pandas as pd
from tqdm import tqdm

from preprocess.core.data_filter import filter_companies
from preprocess.core.data_processor import process_companies, process_benchmarks, process_macro_daily, \
    process_macro_monthly, merging_with_macros, process_factors
from preprocess.core.data_reader import read_companies, read_benchmarks, read_macro_daily, read_macro_monthly, \
    read_factors

ENCODING = 'utf-8'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('date', help="The format of date is YYMMDD(ex. 180131)")
    args = parser.parse_args()
    file_name = args.date

    with tqdm(total=100) as progress_bar:
        progress_bar.set_postfix_str("Reading {}.xlsx...".format(file_name))
        excel_file = pd.ExcelFile('data/{}.xlsx'.format(file_name))
        progress_bar.update(30)

        progress_bar.set_postfix_str("Processing companies...")
        unfiltered_companies = read_companies(excel_file)
        unprocessed_companies = filter_companies(unfiltered_companies)
        processed_companies = process_companies(unprocessed_companies)
        progress_bar.update(30)

        progress_bar.set_postfix_str("Processing macro data...")
        unprocessed_macro_daily = read_macro_daily(excel_file)
        unprocessed_macro_monthly = read_macro_monthly(excel_file)
        processed_macro_daily = process_macro_daily(unprocessed_macro_daily)
        processed_macro_monthly = process_macro_monthly(unprocessed_macro_monthly)
        merged_companies = merging_with_macros(processed_companies, processed_macro_daily,
                                               processed_macro_monthly)
        merged_companies.to_csv('data/{}_company.csv'.format(file_name), index=False, encoding=ENCODING)
        progress_bar.update(20)

        progress_bar.set_postfix_str("Processing benchmarks...")
        unprocessed_benchmarks = read_benchmarks(excel_file)
        processed_benchmarks = process_benchmarks(unprocessed_benchmarks)
        processed_benchmarks.to_csv('data/{}_benchmark.csv'.format(file_name), index=False, encoding=ENCODING)
        progress_bar.update(10)

        progress_bar.set_postfix_str("Processing factors...")
        transposed_factors = read_factors(excel_file)
        processed_factors = process_factors(transposed_factors)
        processed_factors.to_csv('data/{}_factor.csv'.format(file_name), index=False, encoding=ENCODING)
        progress_bar.update(10)


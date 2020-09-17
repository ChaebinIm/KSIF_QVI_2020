# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 6. 19.
"""
import unittest
from unittest import TestCase

import sys
from preprocess.core.columns import *
from preprocess.core.data_filter import filter_companies
from preprocess.core.data_reader import read_companies, read_benchmarks

TEST_NORMAL = '테스트정상'
TEST_HOLDING_COMPANY = '테스트지주사'
TEST_SPAC = '테스트스팩'
TEST_BANKRUPT = '테스트파산'
TEST_MERGE = '테스트합병'
TEST_RELISTED = '테스트재상장'

NUM_COMPANIES = 6
NUM_BENCHMARKS = 11
NUM_DATES = 235

FILE_NAME = 'test_core_data'


class TestDataReader(TestCase):

    def test_read_companies(self):
        companies = read_companies(file_name=FILE_NAME)

        # Check the essential columns.
        self.assertIn(CODE, companies.columns)
        self.assertIn(DATE, companies.columns)
        self.assertIn(NAME, companies.columns)

        # Check the number of companies.
        self.assertEqual(NUM_COMPANIES, len(companies[CODE].unique()))

        # Check the list of companies.
        self.assertListEqual(
            sorted([TEST_NORMAL, TEST_HOLDING_COMPANY, TEST_SPAC, TEST_MERGE, TEST_BANKRUPT, TEST_RELISTED]),
            sorted(list(companies[NAME].unique()))
        )

        # Check all companies have same number of dates.
        self.assertEqual(NUM_DATES, len(companies[DATE].unique()))

    def test_read_benchmarks(self):
        benchmarks = read_benchmarks(file_name=FILE_NAME)

        # Check the essential columns.
        self.assertListEqual([CODE, DATE, PRICE_INDEX], list(benchmarks.columns))

        # Check the number of benchmarks.
        self.assertEqual(NUM_BENCHMARKS, len(benchmarks[CODE].unique()))

        # Check all benchmarks have same number of dates.
        for code in benchmarks[CODE].unique():
            self.assertEqual(NUM_DATES, len(benchmarks.loc[benchmarks[CODE] == code, DATE].unique()))


class TestDataFilter(TestCase):

    def test_filter_companies(self):
        unfiltered_companies = read_companies(file_name=FILE_NAME)
        filtered_companies = filter_companies(unfiltered_companies=unfiltered_companies)
        filtered_company_names = list(filtered_companies[NAME].unique())

        # Check whether TEST_NORMAL is contained or not.
        self.assertIn(TEST_NORMAL, filtered_company_names)

        # Check whether TEST_SPAC is filtered or not.
        self.assertNotIn(TEST_SPAC, filtered_company_names)

        # Check whether the last price of TEST_BANKRUPT is 0 or not.
        test_bankrupt_companies = filtered_companies[filtered_companies[NAME] == TEST_BANKRUPT]
        merge_last_price = test_bankrupt_companies.iloc[-1][ENDP]
        merge_last_adjusted_price = test_bankrupt_companies.iloc[-1][ADJP]
        self.assertEqual(0, merge_last_price)
        self.assertEqual(0, merge_last_adjusted_price)

        # Check whether 2018-01 and 2018-02 are suspended or not.
        test_relisted_companies = filtered_companies.loc[filtered_companies[NAME] == TEST_RELISTED, :]
        self.assertTrue(
            test_relisted_companies.loc[filtered_companies[DATE] == '2018-01-31', :][IS_SUSPENDED].values[0])
        self.assertTrue(
            test_relisted_companies.loc[filtered_companies[DATE] == '2018-02-28', :][IS_SUSPENDED].values[0])

        # Check whether the last price of TEST_MERGE is same with the second last price.
        test_merge_companies = filtered_companies[filtered_companies[NAME] == TEST_MERGE]
        merge_last_price = test_merge_companies.iloc[-1][ENDP]
        merge_second_last_price = test_merge_companies.iloc[-2][ENDP]
        merge_last_adjusted_price = test_merge_companies.iloc[-1][ADJP]
        merge_second_last_adjusted_price = test_merge_companies.iloc[-2][ADJP]
        self.assertIsNotNone(merge_last_price)
        self.assertIsNotNone(merge_last_adjusted_price)
        self.assertEqual(merge_last_price, merge_second_last_price)
        self.assertEqual(merge_last_adjusted_price, merge_second_last_adjusted_price)


class TestDataProcessor(TestCase):

    def test_process_companies(self):
        self.fail()


if __name__ == '__main__':
    # print(sys.path)
    unittest.main()

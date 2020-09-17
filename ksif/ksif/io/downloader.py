# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 7. 18.
"""
import os
from pathlib import Path

import urllib3
from pandas import DataFrame
from pandas import read_hdf

from .google_drive import query_google_spreadsheet, query_google_csv_file, GoogleQueryException, CSV_FILES_GID
from ..core.columns import DATE
from ..util.memoization import memoize
from ..util.retrial import retry

urllib3.disable_warnings()  # Ignore InsecureRequestWarning.

TABLE = 'table'

COMPANY = 'company'
BENCHMARK = 'benchmark'
FACTOR = 'factor'

# CSV encoding type
ENCODING = 'utf-8'

DATA_DIR = 'data'


@memoize(copy=True)
@retry(GoogleQueryException)
def download_latest_data(download_company_data) -> (DataFrame, DataFrame):
    csv_files_url = query_google_spreadsheet(CSV_FILES_GID)
    csv_files_url = csv_files_url.sort_values(by=DATE)
    latest_company_id = csv_files_url.iloc[-1][COMPANY]
    latest_company_file_name = csv_files_url.iloc[-1][DATE] + '_' + COMPANY
    latest_benchmark_id = csv_files_url.iloc[-1][BENCHMARK]
    latest_benchmark_file_name = csv_files_url.iloc[-1][DATE] + '_' + BENCHMARK
    latest_factor_id = csv_files_url.iloc[-1][FACTOR]
    latest_factor_file_name = csv_files_url.iloc[-1][DATE] + '_' + FACTOR

    if not Path(DATA_DIR).exists():
        os.makedirs(DATA_DIR)

    if download_company_data:
        latest_company_data = _download_data(latest_company_file_name, latest_company_id)
    else:
        latest_company_data = None
    latest_benchmark_data = _download_data(latest_benchmark_file_name, latest_benchmark_id)
    latest_factor_data = _download_data(latest_factor_file_name, latest_factor_id)
    latest_factor_data.set_index(DATE, inplace=True)

    return latest_company_data, latest_benchmark_data, latest_factor_data


def _download_data(file_name, id):
    local_company_file_path = '{}/{}.h5'.format(DATA_DIR, file_name)
    if Path(local_company_file_path).exists():
        latest_company_data = custom_read_hdf(local_company_file_path)
    else:
        print("Downloading {} from web...".format(file_name))
        latest_company_data = query_google_csv_file(id)
        latest_company_data.to_hdf(local_company_file_path, key=TABLE, mode='w')
        print("{} is saved as {}.".format(file_name, local_company_file_path))
    return latest_company_data


def custom_read_hdf(path):
    latest_korea_data = read_hdf(path, key=TABLE)
    return latest_korea_data

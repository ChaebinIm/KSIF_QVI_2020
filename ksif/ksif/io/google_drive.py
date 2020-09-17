# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 7. 18.
"""
from io import StringIO

import pandas as pd
from pandas import DataFrame
import requests
import urllib3

from ..errors import GoogleQueryException
from ..core.columns import DATE

urllib3.disable_warnings()  # Ignore InsecureRequestWarning.

SPREADSHEET_DOWNLOAD_REQUEST_URL = 'https://docs.google.com/spreadsheet/ccc?key=1e_7FLT22f0MPfk0zuisbCukWVoCKT3Rm9tPi6R7xntc&output=csv&gid={}'

CSV_FILE_DOWNLOAD_REQUEST_URL = 'https://drive.google.com/uc?export=download'

CSV_FILES_GID = '0'
HOLDING_COMPANY_GID = '1815448628'
DELISTED_COMPANY_GID = '324275710'
MERGED_COMPANY_GID = '1896384832'
RELISTED_COMPANY_GID = '1860868662'


def query_google_spreadsheet(gid) -> DataFrame:
    """
    :param gid: (str or None) 다운받을 시트의 이름.

    :return spreadsheet: (DataFrame)
    """
    if gid is not None and not isinstance(gid, str):
        raise TypeError("Parameter gid should be a None or a string.")

    spreadsheet_download_request_url = SPREADSHEET_DOWNLOAD_REQUEST_URL.format(gid)
    response = requests.get(spreadsheet_download_request_url)

    if response.status_code != 200:
        raise GoogleQueryException(response.status_code)

    spreadsheet = pd.read_csv(StringIO(response.content.decode(encoding='UTF-8', errors='strict')))

    return spreadsheet


def query_google_csv_file(csv_file_id) -> DataFrame:
    """
    :param csv_file_id: (str) 공유가능한 링크에 명시된 ID.
        예를 들어 공유가능한 링크가 https://drive.google.com/open?id=1ee9ZtqUgPAibE7YtN06e1rW5UnhEabjL 일 때,
        1ee9ZtqUgPAibE7YtN06e1rW5UnhEabjL 가 ID.

    :return csv_file: (DataFrame)
    """
    session = requests.Session()

    response = session.get(CSV_FILE_DOWNLOAD_REQUEST_URL, params={'id': csv_file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': csv_file_id, 'confirm': token}
        response = session.get(CSV_FILE_DOWNLOAD_REQUEST_URL, params=params, stream=True)

    csv_file = pd.read_csv(StringIO(response.content.decode(encoding='UTF-8', errors='strict')), parse_dates=[DATE])
    session.close()
    return csv_file


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

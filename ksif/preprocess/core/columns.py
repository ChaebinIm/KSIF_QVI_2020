# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 6. 22.
"""
from ksif.core.columns import *

# Company
# Names of raw excel sheets
COMPANY = 'company'
BENCHMARK = 'benchmark'
MACRO_DAILY = 'macro_daily'
MACRO_MONTHLY = 'macro_monthly'
FACTOR = 'factor'

# Special words for companies
SPAC = '스팩'
HOLDING = 'holding'

# Special words for benchmarks
PRICE_INDEX = 'price_index'

# Common columns
FISCAL_QUARTER = 'fiscal_quarter'

# Daily columns
OUTCST = 'outcst'  # 상장주식수 (보통)(주)
CS_TOBEPUB = 'cs_tobepub'  # 상장예정주식수 (보통)(주)
TRADING_VOLUME = 'trading_volume'  # 거래량 (20일 평균)(주)
NET_PERSONAL_PURCHASE = 'net_personal_purchase'  # 순매수수량(개인)(주)
NET_NATIONAL_PURCHASE = 'net_national_purchase'  # 순매수수량(국가)(주)
NET_FINANCIAL_INVESTMENT_PURCHASE = 'net_financial_investment_purchase'  # 순매수수량(금융투자)(주)
NET_INSTITUTIONAL_FOREIGN_PURCHASE = 'net_institutional_foreign_purchase'  # 순매수수량(기관/외국인계)(주)
NET_INSTITUTIONAL_PURCHASE = 'net_institutional_purchase'  # 순매수수량(기관계)(주)
NET_ETC_FINANCE_PURCHASE = 'net_etc_finance_purchase'  # 순매수수량(기타금융)(주)
NET_ETC_CORPORATION_PURCHASE = 'net_etc_corporation_purchase'  # 순매수수량(기타법인)(주)
NET_ETC_FOREIGN_PURCHASE = 'net_etc_foreign_purchase'  # 순매수수량(기타외국인)(주)
NET_REGISTERED_FOREIGN_PURCHASE = 'net_registered_foreign_purchase'  # 순매수수량(등록외국인)(주)
NET_INSURANCE_PURCHASE = 'net_insurance_purchase'  # 순매수수량(보험)(주)
NET_PRIVATE_FUND_PURCHASE = 'net_private_fund_purchase'  # 순매수수량(사모펀드)(주)
NET_PENSION_PURCHASE = 'net_pension_purchase'  # 순매수수량(연기금)(주)
NET_FOREIGN_PURCHASE = 'net_foreign_purchase'  # 순매수수량(외국인계)(주)
NET_BANK_PURCHASE = 'net_bank_purchase'  # 순매수수량(은행)(주)
NET_TRUST_PURCHASE = 'net_trust_purchase'  # 순매수수량(투신)(주)
SHORT_SALE_VOLUME = 'short_sale_volume'  # 공매도거래량 (20일)(주)
SHORT_SALE_BALANCE = 'short_sale_balance'  # 공매도잔고량(주)
SHARE_LENDING_VOLUME = 'share_lending_volume'  # 대차거래 체결 (20일)(주)
SHARE_LENDING_BALANCE = 'share_lending_balance'  # 대차거래 잔고(주)

DAILY_DATA = [
    CODE, DATE, NAME, FN_GUIDE_SECTOR, FN_GUIDE_INDUSTRY_GROUP_27, ADJP, OUTCST, CS_TOBEPUB, ENDP,
    BETA_1D, BETA_1W, BETA_2W, BETA_3M, BETA_5M, VOL_1D, VOL_1W, VOL_2W, VOL_3M, VOL_5M, IS_MANAGED,
    IS_SUSPENDED, TRADING_VOLUME, EXCHANGE, KRX_SECTOR, HOLDING,
    NET_PERSONAL_PURCHASE, NET_NATIONAL_PURCHASE, NET_FINANCIAL_INVESTMENT_PURCHASE,
    NET_INSTITUTIONAL_FOREIGN_PURCHASE, NET_INSTITUTIONAL_PURCHASE, NET_ETC_FINANCE_PURCHASE,
    NET_ETC_CORPORATION_PURCHASE, NET_ETC_FOREIGN_PURCHASE, NET_REGISTERED_FOREIGN_PURCHASE,
    NET_INSURANCE_PURCHASE, NET_PRIVATE_FUND_PURCHASE, NET_PENSION_PURCHASE, NET_FOREIGN_PURCHASE,
    NET_BANK_PURCHASE, NET_TRUST_PURCHASE, FOREIGN_OWNERSHIP_RATIO, SHORT_SALE_VOLUME, SHORT_SALE_BALANCE,
    SHARE_LENDING_VOLUME, SHARE_LENDING_BALANCE,
]

# Quarterly columns
ASSETS = 'assets'  # 자산(*)(천원)
CUR_ASSETS = 'cur_assets'  # 유동자산(*)(천원)
LIAB = 'liab'  # 부채(*)(천원)
CUR_LIAB = 'cur_liab'  # 유동부채(*)(천원)
INV = 'inv'  # 재고자산(*)(천원)
SALES = 'sales'  # 매출액(수익)(*)(천원)
GP = 'gp'  # 매출총이익(손실)(천원)
NI_OWNER = 'ni_owner'  # 지배주주순이익(천원)
NI = 'net_income'  # 당기순이익(천원)
INT_INC = 'int_inc'  # 이자수익(비영업)(천원)
INT_EXP = 'int_exp'  # 이자비용(비영업)(천원)
CFO = 'cfo'  # 영업활동으로인한현금흐름(천원)
AR = 'ar'  # 매출채권(천원)
ALLOWANCE_AR_ = '(allowance_ar)'  # (대손충당금)(천원)
AP = 'ap'  # 매입채무(천원)
EBIT = 'ebit'  # EBIT(천원)
EBITDA = 'ebitda'  # EBITDA(천원)
CASH = 'cash'  # 현금및현금성자산(*)(천원)
TAX = 'tax'  # 법인세비용(천원)
RES_EXP = 'res_exp'  # 연구개발비(천원)
EQUITY = 'equity'  # 자본(*)(천원)
TANG_ASSET = 'tang_asset'  # 유형자산(*)(천원)
FIN_LIAB = 'fin_liab'  # *총금융부채(천원)

QUARTERLY_DATA = [
    CODE, DATE, ASSETS, CUR_ASSETS, LIAB, CUR_LIAB, INV, SALES, GP, NI_OWNER, NI, INT_INC, INT_EXP, CFO, AR,
    ALLOWANCE_AR_, AP, EBIT, EBITDA, CASH, TAX, RES_EXP, DIVP, EQUITY, TANG_ASSET, FIN_LIAB,
]

COMPANY_RENAMES = {
    'FnGuide Sector': FN_GUIDE_SECTOR,
    'FnGuide Industry Group 27': FN_GUIDE_INDUSTRY_GROUP_27,
    '수정주가 (현금배당반영)(원)': ADJP,
    '상장주식수 (보통)(주)': OUTCST,
    '상장예정주식수 (보통)(주)': CS_TOBEPUB,
    '종가(원)': ENDP,
    '베타 (D,1Yr)': BETA_1D,
    '베타 (W,1Yr)': BETA_1W,
    '베타 (W,2Yr)': BETA_2W,
    '베타 (M,3Yr)': BETA_3M,
    '베타 (M,5Yr)': BETA_5M,
    '변동성 (D,1Yr)': VOL_1D,
    '변동성 (W,1Yr)': VOL_1W,
    '변동성 (W,2Yr)': VOL_2W,
    '변동성 (M,3Yr)': VOL_3M,
    '변동성 (M,5Yr)': VOL_5M,
    '관리종목여부': IS_MANAGED,
    '거래정지여부': IS_SUSPENDED,
    '거래소(시장)': EXCHANGE,
    '자산(*)(천원)': ASSETS,
    '유동자산(*)(천원)': CUR_ASSETS,
    '부채(*)(천원)': LIAB,
    '유동부채(*)(천원)': CUR_LIAB,
    '재고자산(*)(천원)': INV,
    '매출액(수익)(*)(천원)': SALES,
    '매출총이익(손실)(천원)': GP,
    '지배주주순이익(천원)': NI_OWNER,
    '당기순이익(천원)': NI,
    '이자수익(비영업)(천원)': INT_INC,
    '이자비용(비영업)(천원)': INT_EXP,
    '영업활동으로인한현금흐름(천원)': CFO,
    '매출채권(천원)': AR,
    '(대손충당금)(천원)': ALLOWANCE_AR_,
    '매입채무(천원)': AP,
    'EBIT(천원)': EBIT,
    'EBITDA(천원)': EBITDA,
    '현금및현금성자산(*)(천원)': CASH,
    '법인세비용(천원)': TAX,
    '연구개발비(천원)': RES_EXP,
    '배당수익률(보통주,현금)(%)': DIVP,
    '자본(*)(천원)': EQUITY,
    '유형자산(*)(천원)': TANG_ASSET,
    '*총금융부채(천원)': FIN_LIAB,
    '거래소 업종': KRX_SECTOR,
    '순매수수량(개인)(20일합산)(주)': NET_PERSONAL_PURCHASE,
    '순매수수량(기관/외국인계)(20일합산)(주)': NET_INSTITUTIONAL_FOREIGN_PURCHASE,
    '순매수수량(기관계)(20일합산)(주)': NET_INSTITUTIONAL_PURCHASE,
    '순매수수량(금융투자)(20일합산)(주)': NET_FINANCIAL_INVESTMENT_PURCHASE,
    '순매수수량(보험)(20일합산)(주)': NET_INSURANCE_PURCHASE,
    '순매수수량(은행)(20일합산)(주)': NET_BANK_PURCHASE,
    '순매수수량(투신)(20일합산)(주)': NET_TRUST_PURCHASE,
    '순매수수량(사모펀드)(20일합산)(주)': NET_PRIVATE_FUND_PURCHASE,
    '순매수수량(기타금융)(20일합산)(주)': NET_ETC_FINANCE_PURCHASE,
    '순매수수량(연기금)(20일합산)(주)': NET_PENSION_PURCHASE,
    '순매수수량(국가)(20일합산)(주)': NET_NATIONAL_PURCHASE,
    '순매수수량(기타법인)(20일합산)(주)': NET_ETC_CORPORATION_PURCHASE,
    '순매수수량(외국인계)(20일합산)(주)': NET_FOREIGN_PURCHASE,
    '순매수수량(등록외국인)(20일합산)(주)': NET_REGISTERED_FOREIGN_PURCHASE,
    '순매수수량(기타외국인)(20일합산)(주)': NET_ETC_FOREIGN_PURCHASE,
    '외국인보유비중(티커)(%)': FOREIGN_OWNERSHIP_RATIO,
    '공매도거래량 (20일)(주)': SHORT_SALE_VOLUME,
    '공매도잔고량(주)': SHORT_SALE_BALANCE,
    '거래량 (20일 평균)(주)': TRADING_VOLUME,
    '대차거래 체결 (20일)(주)': SHARE_LENDING_VOLUME,
    '대차거래 잔고(주)': SHARE_LENDING_BALANCE,
}

COMPANY_RESULT_COLUMNS = list(
    set(
        itertools.chain(
            # Base information
            BASE_INFORMATION,
            # Quarterly information,
            QUARTERLY_DATA[2:],
            # Filter
            FILTERS,
            COMPANY_FACTORS
        )
    )
)

# Benchmark
BENCHMARK_RESULT_COLUMNS = [CODE, DATE, BENCHMARK_RET_1, BENCHMARK_RET_3, BENCHMARK_RET_6, BENCHMARK_RET_12]

# Factor
BIG_HIGH = 'Size & Book Value(2X3) 대형 - High'
BIG_MEDIUM = 'Size & Book Value(2X3) 대형 - Medium'
BIG_LOW = 'Size & Book Value(2X3) 대형 - Low'
SMALL_HIGH = 'Size & Book Value(2X3) 소형 - High'
SMALL_MEDIUM = 'Size & Book Value(2X3) 소형 - Medium'
SMALL_LOW = 'Size & Book Value(2X3) 소형 - Low'

FACTOR_RET = 'factor_return'

FACTOR_RESULT_COLUMNS = [DATE, SMB, HML]

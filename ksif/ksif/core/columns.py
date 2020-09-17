# -*- coding: utf-8 -*-
"""
:Author: Jaekyung Kim, Sangheum Cho
:Date: 2019. 7. 29.
"""
import itertools

# Base information
DATADATE = 'datadate'
CODE = 'code'
NAME = 'name'
DATE = 'date'
RET_1 = 'return_1'  # 1개월 수익
RET_3 = 'return_3'  # 3개월 수익
RET_6 = 'return_6'  # 6개월 수익
RET_12 = 'return_12'  # 12개월 수익
ADJP = 'adjp'  # 수정종가
ENDP = 'endp'  # 종가
MKTCAP = 'mktcap'  # 시가총액
BASE_INFORMATION = [CODE, NAME, DATE, RET_1, RET_3, RET_6, RET_12, ADJP, ENDP, MKTCAP]

# Fundamental
REVTQ = 'revtq'
REVT12 = 'revt12'
GPQ = 'gpq'
GP12 = 'gp12'
OPQ = 'opq'
OP12 = 'op12'
NIQ = 'niq'
NI12 = 'ni12'
ATQ = 'atq'
ATQ_MEAN4 = 'atq_mean4'
SEQ = 'seq'
SEQ_MEAN4 = 'seq_mean4'
LTQ = 'ltq'

CF_DEP = 'cf_dep'
CFOQ = 'cfoq'
CFO12 = 'cfo12'

A_TANG = 'a_tang'
A_TANG_MEAN12 = 'a_tang_mean12'
A_CUR = 'a_cur'
A_CUR_MEAN12 = 'a_cur_mean12'
L_CUR = 'l_cur'
L_CUR_MEAN12 = 'l_cur_mean12'


# 수익성지표
S_A = 's_a'  # 매출/자산 Sale over Assets
GP_A = 'gp_a'  # 매출총이익/자산 Gross Profit over Assets
OP_A = 'op_a'  # 영업이익/자산 EBIT over Assets
CF_A = 'cf_a'  # 영업활동으로 인한 현금흐름/자산 Cash Flow from operation over Assets
ROA = 'roa'  # 연간 순이익/자산 Return On Assets
ROE = 'roe'  # 연간 순이익/자본 Return On Equity
QROA = 'qroa'  # 분기간 순이익/자산
QROE = 'qroe'  # 분기간 순이익/자본
EBT_E = 'ebt_e'  # (순이익+세금)/자본
ROIC = 'roic'  # 투하자본수익률 Return On Invested Capital
GP_S = 'gp_s'  # 매출총이익/매출
DIVP = 'divp'  # 배당률 Dividend rate(12 month rolling)
PROFIT_FACTORS = [S_A, GP_A, OP_A, CF_A, ROA, ROE, QROA, QROE, EBT_E, ROIC, GP_S, DIVP]




# Filter
EXCHANGE = 'exchange'  # 거래소
FN_GUIDE_SECTOR = 'fn_guide_sector'  # FN Guide 섹트
FN_GUIDE_INDUSTRY_GROUP_27 = 'fn_guide_industry_group_27'  # FN Guide 산업구분
KRX_SECTOR = 'krx_sector'  # 한국거래소 섹터
HOLDING = 'holding'  # 지주사 여부
IS_MANAGED = 'is_managed'  # 관리종목 여부
IS_SUSPENDED = 'is_suspended'  # 거래정지 여부
FILTERS = [EXCHANGE, FN_GUIDE_SECTOR, FN_GUIDE_INDUSTRY_GROUP_27, KRX_SECTOR, HOLDING, IS_MANAGED,
           IS_SUSPENDED]

# Value factors
PER = 'per'  # 주가수익률 Per Earnings Ratio
PBR = 'pbr'  # 주가순자산비율 Price to Book-value Ratio
PSR = 'psr'  # 주가매출비율 Price to Sales Ratio
PCR = 'pcr'  # 주가현금흐름비율 Price to Cash flow Ratio
PGPR = 'pgpr'  # 주가매출총이익비율 Price to Gross Profit Ratio
POPR = 'popr'  # 주가영업이익비율 Price to Operating Profit Ratio
EV_EBITDA = 'ev_ebitda'  # 시장가격/세금이자지급전이익
EBIT_EV = 'ebit_ev'  # 영업이익/시장가격
CF_EV = 'cf_ev'  # 영업현금흐름/시장가격
S_EV = 's_ev'  # 매출/시장가격
E_P = 'e_p'  # 1/PER
B_P = 'b_p'  # 1/PBR
C_P = 'c_p'  # 1/PCR
S_P = 's_p'  # 1/PSR
GP_P = 'gp_p'  # 1/PGPR
OP_P = 'op_p'  # 1/POPR
VALUE_FACTORS = [PER, PBR, PSR, PCR, PGPR, POPR, EV_EBITDA, EBIT_EV, CF_EV, S_EV, E_P, B_P, C_P, S_P, GP_P, OP_P]

# market
PRCC = 'prcc'
RET = 'ret'

# Growth factors
SALESQOQ = 'salesqoq'  # 분기간 매출 변화율
GPQOQ = 'gpqoq'  # 분기간 매출총이익 변화율
OPQOQ = 'opqoq'  # 분기간 영업이익 변화율
ROAQOQ = 'roaqoq'  # 분기간 ROA 변화율
ROAYOY = 'roayoy'  # 연간 ROA 변화율
GP_SYOY = 'gp_syoy'  # 연간 매출총이익/매출 변화율
GP_AYOY = 'gp_ayoy'  # 연간 매출총이익/자산 변화율
ASSETSYOY = 'assetsyoy'  # 연간 자산 변화율
ASSETSQOQ = 'assetsqoq'  # 분기간 자산 변화율
GROWTH_FACTORS = [SALESQOQ, GPQOQ, OPQOQ, ROAQOQ, ROAYOY, GP_SYOY, GP_AYOY, ASSETSYOY, ASSETSQOQ]

# Momentum factors
MOM12_1 = 'mom12-1'  # 12개월 모멘텀 - 1개월 모멘텀
MOM6_1 = 'mom6-1'  # 6개월 모멘텀 - 1개월 모멘텀
MOM12 = 'mom12'  # 12개월 모멘텀
MOM6 = 'mom6'  # 6개월 모멘텀
MOM3 = 'mom3'  # 3개월 모멘텀
MOM1 = 'mom1'  # 1개월 모멘텀
MOMENTUM_FACTORS = [MOM12_1, MOM6_1, MOM12, MOM6, MOM3, MOM1]

# Safety factors
BETA_1D = 'beta_1d'  # 일일 베타
BETA_3M = 'beta_3m'  # 3개월 베타
BETA_5M = 'beta_5m'  # 5개월 베타
BETA_1W = 'beta_1w'  # 1주 베타
BETA_2W = 'beta_2w'  # 2주 베타
VOL_1D = 'vol_1d'  # 일일 변동성
VOL_3M = 'vol_3m'  # 3개월 변동성
VOL_5M = 'vol_5m'  # 5개월 변동성
VOL_1W = 'vol_1w'  # 1주 변동성
VOL_2W = 'vol_2w'  # 2주 변동성
LIQ_RATIO = 'liq_ratio'  # 유동비율
DEBT_RATIO = 'debt_ratio'  # 부채비율
EQUITY_RATIO = 'equity_ratio'  # 자본비율
SAFETY_FACTORS = [BETA_1D, BETA_1W, BETA_2W, BETA_3M, BETA_5M, VOL_1D, VOL_1W, VOL_2W, VOL_3M, VOL_5M, LIQ_RATIO,
                  DEBT_RATIO, EQUITY_RATIO]

# Liquidity factors
TRADING_VOLUME_RATIO = 'trading_volume_ratio'  # 거래비율
NET_PERSONAL_PURCHASE_RATIO = 'net_personal_purchase_ratio'  # 개인순매수비율
NET_INSTITUTIONAL_FOREIGN_PURCHASE_RATIO = 'net_institutional_foreign_purchase_ratio'  # 기관/외국인순매수비율
NET_INSTITUTIONAL_PURCHASE_RATIO = 'net_institutional_purchase_ratio'  # 기관순매수비율
NET_FINANCIAL_INVESTMENT_PURCHASE_RATIO = 'net_financial_investment_purchase_ratio'  # 금융투자순매수비율
NET_INSURANCE_PURCHASE_RATIO = 'net_insurance_purchase_ratio'  # 보험순매수비율
NET_TRUST_PURCHASE_RATIO = 'net_trust_purchase_ratio'  # 투신순매수비율
NET_PRIVATE_FUND_PURCHASE_RATIO = 'net_private_fund_purchase_ratio'  # 사모펀드순매수비율
NET_BANK_PURCHASE_RATIO = 'net_bank_purchase_ratio'  # 은행순매수비율
NET_ETC_FINANCE_PURCHASE_RATIO = 'net_etc_finance_purchase_ratio'  # 기타금융순매수비율
NET_PENSION_PURCHASE_RATIO = 'net_pension_purchase_ratio'  # 연기금순매수비율
NET_NATIONAL_PURCHASE_RATIO = 'net_national_purchase_ratio'  # 국가순매수비율
NET_ETC_CORPORATION_PURCHASE_RATIO = 'net_etc_corporation_purchase_ratio'  # 기타법인순매수비율
NET_FOREIGN_PURCHASE_RATIO = 'net_foreign_purchase_ratio'  # 외국인순매수비율
NET_REGISTERED_FOREIGN_PURCHASE_RATIO = 'net_registered_foreign_purchase_ratio'  # 등록외국인순매수비율
NET_ETC_FOREIGN_PURCHASE_RATIO = 'net_etc_foreign_purchase_ratio'  # 기타외국인순매수비율
FOREIGN_OWNERSHIP_RATIO = 'foreign'  # 외국인보유비중
SHORT_SALE_VOLUME_RATIO = 'short_sale_volume_ratio'  # 공매도거래비율
SHORT_SALE_BALANCE_RATIO = 'short_sale_balance_ratio'  # 공매도잔고비율
SHORT_SALE_BALANCE_MOM = 'short_sale_balance_mom'  # 월간 공매도잔고변화율
SHARE_LENDING_VOLUME_RATIO = 'share_lending_volume_ratio'  # 대차거래비율
SHARE_LENDING_BALANCE_RATIO = 'share_lending_balance_ratio'  # 대차잔고비율
SHARE_LENDING_BALANCE_MOM = 'share_lending_balance_mom'  # 월간 공매도잔고변화율
LIQUIDITY_FACTORS = [
    TRADING_VOLUME_RATIO, NET_PERSONAL_PURCHASE_RATIO, NET_INSTITUTIONAL_FOREIGN_PURCHASE_RATIO,
    NET_INSTITUTIONAL_PURCHASE_RATIO, NET_FINANCIAL_INVESTMENT_PURCHASE_RATIO, NET_INSURANCE_PURCHASE_RATIO,
    NET_TRUST_PURCHASE_RATIO, NET_PRIVATE_FUND_PURCHASE_RATIO, NET_BANK_PURCHASE_RATIO,
    NET_ETC_FINANCE_PURCHASE_RATIO, NET_PENSION_PURCHASE_RATIO, NET_NATIONAL_PURCHASE_RATIO,
    NET_ETC_CORPORATION_PURCHASE_RATIO, NET_FOREIGN_PURCHASE_RATIO, NET_REGISTERED_FOREIGN_PURCHASE_RATIO,
    NET_ETC_FOREIGN_PURCHASE_RATIO, FOREIGN_OWNERSHIP_RATIO, SHORT_SALE_VOLUME_RATIO, SHORT_SALE_BALANCE_RATIO,
    SHORT_SALE_BALANCE_MOM, SHARE_LENDING_VOLUME_RATIO, SHARE_LENDING_BALANCE_RATIO, SHARE_LENDING_BALANCE_MOM,
]

COMPANY_FACTORS = list(
    itertools.chain(
        # Values factors
        VALUE_FACTORS,
        # Profit factors
        PROFIT_FACTORS,
        # Growth factors
        GROWTH_FACTORS,
        # Momentum factors
        MOMENTUM_FACTORS,
        # Safety factors
        SAFETY_FACTORS,
        # Liquidity factors
        LIQUIDITY_FACTORS,
    )
)

# Macro factors
TERM_SPREAD_KOR = "term_spread_kor"  # 장단기 국채 금리차이 10 - 1, 한국 국채; 장기 경제 전망
TERM_SPREAD_US = "term_spread_us"  # 장단기 국채 금리차이 10 - 1, 미국 국채
CREDIT_SPREAD_KOR = "credit_spread_kor"  # 회사채 BBB- - AA- ; 신용위험도
LOG_USD2KRW = "log_usd2krw"  # 1달러 당 원화 가격의 로그
LOG_CHY2KRW = "log_chy2krw"  # 1위안 당 원화 가격의 로그
LOG_EURO2KRW = "log_euro2krw"  # 1유로 당 원화 가격의 로그
TED_SPREAD = "ted_spread"  # 리보금리 - 미국 국채 스프레드 (1개월물) ; 미국 역외 달러 조달의 위험성 ; 무역과 internal banking system의 안정성 지표
LOG_NYSE = "log_nyse"  # 뉴욕증권거래소 지수의 로그
LOG_NASDAQ = "log_nasdaq"  # 나스닥 거래소 지수의 로그
LOG_SEMI_CONDUCTOR = "log_semi_conductor"  # 반도체가격의 로그
LOG_DOLLAR_INDEX = "log_dollar_index"  # 달러 인덱스의 로그; 달러가 다른 통화대비 얼마나 강세인지 보여줌
LOG_OIL = "log_oil"  # 유가의 로그
LOG_EXPORT = "log_export"  # 한국의 수출량 로그
LOG_IMPORT = "log_import"  # 한국의 수입량 로그
LOG_INDUSTRY_PRODUCTION_US = "log_industry_production_us"  # 미국의 산업생산 로그
LOG_INDUSTRY_PRODUCTION_EURO = "log_industry_production_euro"  # 유로지역의 산업생산 로그
LOG_INDUSTRY_PRODUCTION_KOR = "log_industry_production_kor"  # 한국의 산업생산 로그

MACRO_FACTORS = [
    TERM_SPREAD_KOR, TERM_SPREAD_US, CREDIT_SPREAD_KOR, LOG_USD2KRW, LOG_CHY2KRW, LOG_EURO2KRW, TED_SPREAD, LOG_NYSE,
    LOG_NASDAQ, LOG_SEMI_CONDUCTOR, LOG_DOLLAR_INDEX, LOG_OIL,
]

# Benchmarks
MKF_2000 = 'MKF2000'
KOSDAQ = '코스닥'
KOSDAQ_LARGE = '코스닥 대형주'
KOSDAQ_MIDDLE = '코스닥 중형주'
KOSDAQ_SMALL = '코스닥 소형주'
KOSPI = '코스피'
KOSPI_200 = '코스피 200'
KOSPI_LARGE = '코스피 대형주'
KOSPI_MIDDLE = '코스피 중형주'
KOSPI_SMALL = '코스피 소형주'
TOTAL_LARGE = '코스피/코스닥 대형주 평균'
TOTAL_MIDDLE = '코스피/코스닥 중형주 평균'
TOTAL_SMALL = '코스피/코스닥 소형주 평균'
CD91 = 'CD 91일'

BENCHMARK_RET_1 = 'benchmark_return_1'
BENCHMARK_RET_3 = 'benchmark_return_3'
BENCHMARK_RET_6 = 'benchmark_return_6'
BENCHMARK_RET_12 = 'benchmark_return_12'

BENCHMARKS = [
    MKF_2000,
    KOSDAQ, KOSDAQ_LARGE, KOSDAQ_MIDDLE, KOSDAQ_SMALL,
    KOSPI, KOSPI_200, KOSPI_LARGE, KOSPI_MIDDLE, KOSPI_SMALL,
    TOTAL_LARGE, TOTAL_MIDDLE, TOTAL_SMALL,
    CD91,
]

# Factors
SMB = 'SMB'
HML = 'HML'
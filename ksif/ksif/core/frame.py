# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
         Park Ji woo
:Date: 2018. 7. 18
"""
import os
import platform
import sys
from copy import deepcopy as dc
from datetime import datetime
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.core.common as com
import statsmodels.api as sm
from matplotlib import font_manager, rc
from pandas import DataFrame
from pandas import Series
from pandas.core.index import MultiIndex
from pandas.core.indexing import convert_to_index_sliceable
from performanceanalytics.charts.performance_summary import create_performance_summary

from .columns import *
from .outcomes import *
from ..io.downloader import download_latest_data
from ..util.checker import not_empty
import dropbox
import io

# Hangul font setting
# noinspection PyProtectedMember
font_manager._rebuild()
if platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()
elif platform.system() == 'Darwin':  # OS X
    font_name = font_manager.FontProperties(fname='/Library/Fonts/AppleGothic.ttf').get_name()
else:  # Linux
    fname = '/usr/share/fonts/truetype/nanum/NanumGothicOTF.ttf'
    if not os.path.isfile(fname):
        raise ResourceWarning("Please install NanumGothicOTF.ttf for plotting Hangul.")
    font_name = font_manager.FontProperties(fname=fname).get_name()
rc('font', family=font_name)

# for fix broken Minus sign
matplotlib.rcParams['axes.unicode_minus'] = False

PERCENTAGE = 'percentage'

WEIGHT = 'weight'
WEIGHT_SUM = 'weight_sum'

START_DATE = datetime(year=2001, month=5, day=31)

QUANTILE = 'quantile'
RANK = 'rank'
RANK_CORRELATION = 'Rank correlation'


class Portfolio(DataFrame):
    """

    """
    _benchmark = KOSPI
    benchmarks = None
    factors = None

    @property
    def _constructor(self):
        return Portfolio

    @not_empty
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy: bool = False,
                 start_date: datetime = START_DATE, end_date: datetime = None,
                 include_holding: bool = False, include_finance: bool = False,
                 include_managed: bool = False, include_suspended: bool = False):

        if not end_date:
            end_date = datetime.today()

        if data is None:
            print('Data is being downloaded from KSIF DROPBOX DATA STORAGE')
            dbx = dropbox.Dropbox(
                oauth2_access_token='TVRotgoEpxAAAAAAAAAAMyxLV0OXl61S_mXAzvj7tynmAbUz6J2mgIDYvAh-XxHG', timeout=None)

            metadata, f = dbx.files_download('/preprocessed/final_msf.csv')
            # metadata, f = dbx.files_download('/preprocessed/merged.csv')
            binary_file = f.content
            data = pd.read_csv(io.BytesIO(binary_file))

            #
            _, self.benchmarks, self.factors = download_latest_data(download_company_data=False)
            #
            # if not include_holding:
            #     data = data.loc[~data[HOLDING], :]
            #
            # if not include_finance:
            #     data = data.loc[data[FN_GUIDE_SECTOR] != '금융', :]
            #
            # if not include_managed:
            #     data = data.loc[~data[IS_MANAGED], :]
            #
            # if not include_suspended:
            #     data = data.loc[~data[IS_SUSPENDED], :]
            #
            # data = data.loc[(start_date <= data[DATE]) & (data[DATE] <= end_date), :]

        else:
            _, self.benchmarks, self.factors = download_latest_data(download_company_data=False)

        self.benchmarks = self.benchmarks.loc[
                          (start_date <= self.benchmarks[DATE]) & (self.benchmarks[DATE] <= end_date), :]
        self.factors = self.factors.loc[(start_date <= self.factors.index) & (self.factors.index <= end_date), :]

        super(Portfolio, self).__init__(data=data) #, index=index, columns=columns, dtype=dtype, copy=copy)
        # self.data = data

    def __getitem__(self, key):
        from pandas.core.dtypes.common import is_list_like, is_integer, is_iterator

        key = com.apply_if_callable(key, self)

        # shortcut if the key is in columns
        try:
            if self.columns.is_unique and key in self.columns:
                if self.columns.nlevels > 1:
                    return self._getitem_multilevel(key)
                return self._get_item_cache(key)
        except (TypeError, ValueError):
            # The TypeError correctly catches non hashable "key" (e.g. list)
            # The ValueError can be removed once GH #21729 is fixed
            pass

        # Do we have a slicer (on rows)?
        indexer = convert_to_index_sliceable(self, key)
        if indexer is not None:
            return self._slice(indexer, axis=0)

        # Do we have a (boolean) DataFrame?
        if isinstance(key, DataFrame):
            return self._getitem_frame(key)

        # Do we have a (boolean) 1d indexer?
        if com.is_bool_indexer(key):
            return self._getitem_bool_array(key)

        # We are left with two options: a single key, and a collection of keys,
        # We interpret tuples as collections only for non-MultiIndex
        is_single_key = isinstance(key, tuple) or not is_list_like(key)

        if is_single_key:
            if self.columns.nlevels > 1:
                return self._getitem_multilevel(key)
            indexer = self.columns.get_loc(key)
            if is_integer(indexer):
                indexer = [indexer]
        else:
            if is_iterator(key):
                key = list(key)
            # noinspection PyProtectedMember
            indexer = self.loc._convert_to_indexer(key, axis=1, raise_missing=True)

        # take() does not accept boolean indexers
        if getattr(indexer, "dtype", None) == bool:
            indexer = np.where(indexer)[0]

        data = self._take(indexer, axis=1)

        if is_single_key:
            # What does looking for a single key in a non-unique index return?
            # The behavior is inconsistent. It returns a Series, except when
            # - the key itself is repeated (test on data.shape, #9519), or
            # - we have a MultiIndex on columns (test on self.columns, #21309)
            if data.shape[1] == 1 and not isinstance(self.columns, MultiIndex):
                data = data[key]

        return data

    @property
    def benchmark(self):
        return self._benchmark

    @not_empty
    def get_benchmark(self, benchmark: str = None) -> DataFrame:
        """
        Return a benchmark of this portfolio period.

        :param benchmark: (str) The name of benchmark to use. If benchmark is None, use self.benchmark.

        :return selected_benchmark: (DataFrame)
            code                | (str) The name of benchmark. ex) 코스피, 코스닥...
            date                | (datetime)
            benchmark_return_1  | (float) 1 month return.
            benchmark_return_3  | (float) 3 month return.
            benchmark_return_6  | (float) 6 month return.
            benchmark_return_12 | (float) 12 month return.
        """
        if benchmark is not None and benchmark not in BENCHMARKS:
            raise ValueError('{} is not registered.'.format(benchmark))

        if benchmark:
            selected_benchmark = self.benchmarks.loc[(self.benchmarks[CODE] == benchmark) &
                                                     (self.benchmarks[DATE] >= min(self[DATE])) &
                                                     (self.benchmarks[DATE] <= max(self[DATE])), :]
        else:
            selected_benchmark = self.benchmarks.loc[(self.benchmarks[CODE] == self._benchmark) &
                                                     (self.benchmarks[DATE] >= min(self[DATE])) &
                                                     (self.benchmarks[DATE] <= max(self[DATE])), :]

        selected_benchmark.set_index(DATE, inplace=True)
        selected_benchmark = selected_benchmark.loc[
                             :, [BENCHMARK_RET_1, BENCHMARK_RET_3, BENCHMARK_RET_6, BENCHMARK_RET_12]
                             ]
        return selected_benchmark

    def set_benchmark(self, benchmark):
        if benchmark not in BENCHMARKS:
            raise ValueError('{} is not registered.'.format(benchmark))
        else:
            self._benchmark = benchmark

    # noinspection PyPep8Naming
    @property
    def SMB(self) -> Series:
        return self.factors.loc[:, SMB]

    # noinspection PyPep8Naming
    @property
    def HML(self) -> Series:
        return self.factors.loc[:, HML]

    @not_empty
    def to_dataframe(self, deepcopy: bool = True) -> DataFrame:
        """
        Convert portfolio to dataframe type.

        :param deepcopy : (bool) If deepcopy is True, convert to dataframe based on deepcopy. Or, convert to dataframe
                                  based on shallow copy.

        :return dataframe : (DataFrame) Converted dataframe type portfolio
        """
        if deepcopy:
            dataframe = DataFrame(dc(self))
        else:
            dataframe = DataFrame(self)

        return dataframe

    def outcome(self, benchmark: str = None, weighted: str = None,
                long_transaction_cost_ratio: float = 0.0025, short_transaction_cost_ratio: float = 0.0025,
                show_plot: bool = False):
        """
        Calculate various indices of the portfolio.

        :param benchmark: (str) The name of benchmark. If benchmark is None, use a default benchmark.
        :param weighted: (str) If weighted is a string, use the string to calculate weighted portfolio.
                                If there are negative weights, calculate long-short weighted portfolio.
        :param long_transaction_cost_ratio: (float) A transaction cost ratio for long investment
        :param short_transaction_cost_ratio: (float) A transaction cost ratio for short investment
        :param show_plot: (bool) If show_plot is True, show a performance summary graph.

        :return result: (dict)
            portfolio_return            | (float) Total compound return of the portfolio
            benchmark_return            | (float) Total compound return of the benchmark
            active_return               | (float) Annualized average excess return
            active_risk                 | (float) Annualized tracking error
            sharpe_ratio                | (float) Sharpe ratio
            information_ratio           | (float) Average excess return / tracking error
            compound_annual_growth_rate | (float) Annual compound return of the portfolio
            maximum_drawdown            | (float) The maximum loss from a peak to a trough of a portfolio,
                                                  before a new peak is attained
            Fama_French_alpha           | (float) An abnormal return from Fama-French 3 Factor model
                                                  (Fama and French, 1993)
            Fama_French_alpha_p_value   | (float) A p-value of the the abnormal return
            Fama_French_beta            | (float) A market beta from Fama-French 3 Factor model
                                                  (Fama and French, 1993)
            turnover                    | (float) Annual average turnover
        """
        if benchmark is not None and benchmark not in BENCHMARKS:
            raise ValueError('{} is not registered.'.format(benchmark))

        if benchmark is None:
            benchmark = self._benchmark

        portfolio, portfolio_returns, turnovers = self.get_returns_and_turnovers(long_transaction_cost_ratio,
                                                                                 short_transaction_cost_ratio, weighted)

        turnover = turnovers.mean() * 12

        benchmarks = portfolio.get_benchmark(benchmark=benchmark).loc[:, [BENCHMARK_RET_1]]
        merged_returns = pd.merge(portfolio_returns, benchmarks, on=DATE)
        merged_returns = pd.merge(merged_returns,
                                  portfolio.get_benchmark(CD91).rename(columns={BENCHMARK_RET_1: CD91}).loc[:, [CD91]],
                                  on=DATE)

        # Portfolio return, benchmark return
        portfolio_return = self._calculate_total_return(merged_returns[PORTFOLIO_RETURN])
        benchmark_return = self._calculate_total_return(merged_returns[BENCHMARK_RET_1])

        # CAGR
        period_len = len(portfolio[DATE].unique())
        compound_annual_growth_rate = (portfolio_return + 1) ** (12 / period_len) - 1

        # Active return, active risk, information ratio
        benchmark_excess_returns = merged_returns[PORTFOLIO_RETURN] - merged_returns[BENCHMARK_RET_1]

        average_excess_return = np.average(benchmark_excess_returns)
        tracking_error = np.std(benchmark_excess_returns)

        active_return = average_excess_return * 12
        active_risk = tracking_error * np.sqrt(12)
        information_ratio = average_excess_return / tracking_error

        # Sharpe ratio
        risk_free_excess_returns = merged_returns[PORTFOLIO_RETURN] - merged_returns[CD91]
        sharpe_ratio = np.average(risk_free_excess_returns) / np.std(risk_free_excess_returns)

        # Maximum drawdown
        portfolio_cumulative_assets = merged_returns[PORTFOLIO_RETURN].add(1).cumprod()
        maximum_drawdown = portfolio_cumulative_assets.div(portfolio_cumulative_assets.cummax()).sub(1).min()

        # Fama-French, 1993
        market_excess_returns = merged_returns[BENCHMARK_RET_1] - merged_returns[CD91]
        risk_free_excess_return = 'risk_free_excess_return'
        market_excess_return = 'market_excess_return'
        ff_data = pd.concat([
            DataFrame(risk_free_excess_returns, columns=[risk_free_excess_return]),
            DataFrame(market_excess_returns, columns=[market_excess_return]),
            self.factors
        ], axis=1, join='inner').dropna()
        model = sm.OLS(
            ff_data.loc[:, risk_free_excess_return],
            sm.add_constant(ff_data.loc[:, [market_excess_return, SMB, HML]])
        ).fit()
        fama_french_alpha = model.params[0]
        fama_french_alpha_p_value = model.pvalues[0]
        fama_french_beta = model.params[1]

        result = {
            PORTFOLIO_RETURN: portfolio_return,
            BENCHMARK_RETURN: benchmark_return,
            ACTIVE_RETURN: active_return,
            ACTIVE_RISK: active_risk,
            SR: sharpe_ratio,
            IR: information_ratio,
            CAGR: compound_annual_growth_rate,
            MDD: maximum_drawdown,
            FAMA_FRENCH_ALPHA: fama_french_alpha,
            FAMA_FRENCH_ALPHA_P_VALUE: fama_french_alpha_p_value,
            FAMA_FRENCH_BETA: fama_french_beta,
            TURNOVER: turnover,
        }

        if show_plot:
            plotting_returns = dc(merged_returns).loc[:, [PORTFOLIO_RETURN, BENCHMARK_RET_1]]
            plotting_returns.rename(columns={
                PORTFOLIO_RETURN: 'Portfolio',
                BENCHMARK_RET_1: benchmark
            }, inplace=True)
            create_performance_summary(plotting_returns, other_cols=range(1, 2))
            plt.show()

        return result

    def get_returns(self, weighted: str = None,
                    long_transaction_cost_ratio: float = 0.0025,
                    short_transaction_cost_ratio: float = 0.0025, cumulative=False) -> DataFrame:
        _, returns, _ = self.get_returns_and_turnovers(
            long_transaction_cost_ratio, short_transaction_cost_ratio, weighted
        )
        if cumulative:
            returns = _cumulate(returns)
        return returns

    def get_returns_and_turnovers(self, long_transaction_cost_ratio, short_transaction_cost_ratio, weighted):
        portfolio = self.dropna(subset=[RET_1])
        returns = pd.DataFrame()
        if weighted:
            if weighted not in self.columns:
                raise ValueError('{} is not in Portfolio.columns.'.format(weighted))
            portfolio = portfolio.dropna(subset=[weighted])
            long_portfolio = portfolio.loc[portfolio[weighted] > 0, :]
            short_portfolio = portfolio.loc[portfolio[weighted] < 0, :]
            short_portfolio.loc[:, RET_1] = -1 * short_portfolio.loc[:, RET_1]
            short_portfolio.loc[:, weighted] = -short_portfolio.loc[:, weighted]
            long_returns = long_portfolio.groupby([DATE]).apply(
                lambda x: np.average(x[RET_1], weights=x[weighted])
            )
            short_returns = short_portfolio.groupby([DATE]).apply(
                lambda x: np.average(x[RET_1], weights=x[weighted])
            )
            long_turnovers = _get_turnovers(long_portfolio, weighted)
            short_turnovers = _get_turnovers(short_portfolio, weighted)
            if short_returns.empty:
                returns[PORTFOLIO_RETURN] = long_returns.subtract(
                    long_turnovers.multiply(long_transaction_cost_ratio), fill_value=0)
                turnovers = long_turnovers
            else:
                returns[PORTFOLIO_RETURN] = long_returns.subtract(
                    long_turnovers.multiply(long_transaction_cost_ratio), fill_value=0
                ).add(
                    short_returns.subtract(
                        short_turnovers.multiply(short_transaction_cost_ratio), fill_value=0)
                )
                if pd.isna(returns[PORTFOLIO_RETURN]).any():
                    warn("When calculating long-short portfolio, weighted should have positive and negative values "
                         "in same periods. Otherwise, the return of the period is not calculated.")
                    returns.dropna(inplace=True)
                turnovers = long_turnovers.add(short_turnovers)
        else:
            turnovers = _get_turnovers(portfolio)
            returns[PORTFOLIO_RETURN] = portfolio.groupby([DATE]).apply(
                lambda x: np.average(x[RET_1])
            ).subtract(turnovers.multiply(short_transaction_cost_ratio), fill_value=0)
        return portfolio, returns, turnovers

    @not_empty
    def _calculate_total_return(self, grouped_data):
        data = grouped_data.dropna()
        total_return = _cumulate(data).iloc[-1]
        return total_return

    @not_empty
    def periodic_rank(self, min_rank: int = 1, max_rank: int = sys.maxsize, factor: str = MKTCAP,
                      bottom: bool = False, drop_rank: bool = True):
        """
        Select companies which have a rank bigger than or equal to min_rank, and smaller than or equal to max_rank
        for each period.

        :param min_rank: (int) The minimum rank of selected companies.
                               The selected_companies includes the minimum ranked company.
        :param max_rank: (int) The maximum rank of selected companies.
                               The selected_companies includes the maximum ranked company.
        :param factor: (str) The factor used to determine rank.
        :param bottom: (bool) If bottom is True, select the companies from bottom. Or, select the companies from top.
        :param drop_rank: (bool) If drop_rank is True, delete rank column from the selected_companies.

        :return selected_companies: (DataFrame) Selected companies for each period by rank of the factor.
        """
        assert min_rank > 0, "min_rank should be bigger than 0."
        assert max_rank > min_rank, "max_rank should be bigger than min_rank."

        all_companies = dc(self)
        all_companies = all_companies.dropna(subset=[factor])
        all_companies[RANK] = all_companies.groupby(by=[DATE])[factor].transform(
            lambda x: x.rank(ascending=bottom)
        )
        selected_companies = all_companies.loc[(all_companies[RANK] >= min_rank) & (all_companies[RANK] <= max_rank), :]
        selected_companies = selected_companies.sort_values(by=[DATE, RANK])

        if drop_rank:
            del selected_companies[RANK]

        return selected_companies

    @not_empty
    def periodic_percentage(self, min_percentage: float, max_percentage: float, factor: str = MKTCAP,
                            bottom: bool = False):
        """
        Select companies which have a percentage bigger than or equal to min_percentage, and smaller than or equal to
        max_percentage for each period.

        :param min_percentage: (float) The minimum percentage of selected companies.
                               The selected_companies includes the minimum percent company.
        :param max_percentage: (float) The maximum percentage of selected companies.
                               The selected_companies does not include the maximum percent company.
        :param factor: (str) The factor used to determine rank.
        :param bottom: (bool) If bottom is True, select the companies from bottom. Or, select the companies from top.

        :return selected_companies: (DataFrame) Selected companies for each period by quantile of the factor.
        """
        assert min_percentage >= 0, "min_percentage should be bigger than or equal to 0."
        assert max_percentage > min_percentage, "max_percentage should be bigger than min_percentage."
        assert max_percentage <= 1, "max_percentage should be smaller than or equal to 0."

        all_companies = dc(self)
        all_companies = all_companies.dropna(subset=[factor])
        all_companies[PERCENTAGE] = all_companies.groupby(by=[DATE])[factor].transform(
            lambda x: x.rank(ascending=bottom, pct=True)
        )
        selected_companies = all_companies.loc[
                             (all_companies[PERCENTAGE] >= min_percentage) &
                             (all_companies[PERCENTAGE] < max_percentage), :]

        del selected_companies[PERCENTAGE]

        return selected_companies

    @not_empty
    def periodic_standardize(self, factor: str, prefix: str = 'std_'):
        """
        Standardize a factor periodically.

        :param factor: (str) The name of factor will be standardized.
        :param prefix: (str) The prefix preceding a name of standardized factor.

        :return standardized_companies: (DataFrame) Standardized companies for each period by factor.
        """
        unstandardized_companies = dc(self.loc[~np.isnan(self[factor]), :])
        unstandardized_companies[prefix + factor] = unstandardized_companies.groupby(by=[DATE])[factor].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        standardized_companies = unstandardized_companies

        return standardized_companies

    @not_empty
    def quantile_distribution_ratio(self, factor: str, chunk_num: int = 10, cumulative: bool = True,
                                    weighted: bool = False, only_positive: bool = False, show_plot: bool = False,
                                    show_bar_chart: bool = False, title: str = None) -> DataFrame:
        """
        Make quantile portfolios by the given factor, and calculate returns.

        :param factor: (str) The name of factor used to make quantile portfolios.
        :param chunk_num: (int) The number of portfolios.
        :param cumulative: (bool) If cumulative is true, calculate cumulative returns.
        :param weighted: (bool) If weighted is true, each portfolio is a weighted portfolio based on MKTCAP
        :param only_positive: (bool) If only_positive is true, use only positive value of the factor.
        :param show_plot: (bool) If show_plot is true, show a time series line chart of groups.
        :param show_bar_chart: (bool) If show_bar_chart is true, show a arithmetic average bar chart of groups.
        :param title: (str) If title is not None, set the title.

        :return quantile_portfolio_returns: (DataFrame) The returns of each group
            --------------------------------------------------------------
            date    | (datetime)
            --------------------------------------------------------------
            1       | (float) The return of group 1 portfolio at the date.
            2       | (float) The return of group 2 portfolio at the date.
            3       | (float) The return of group 3 portfolio at the date.
            ...
            --------------------------------------------------------------
        """
        assert chunk_num > 1, "chunk_num should be bigger than 1."

        labels = [str(x) for x in range(1, chunk_num + 1)]

        portfolio = dc(self)
        portfolio = portfolio.dropna(subset=[factor, RET_1])

        if only_positive:
            portfolio = portfolio.loc[portfolio[factor] > 0, :]

        portfolio[QUANTILE] = portfolio.groupby(by=[DATE])[factor].transform(
            lambda x: pd.qcut(x, chunk_num, labels=labels, duplicates='drop')
        )
        portfolio[QUANTILE] = portfolio[QUANTILE].apply(int).apply(str)

        quantile_portfolio_returns = DataFrame()
        for label in labels:
            labelled_data = portfolio.loc[portfolio[QUANTILE] == label, :]
            if weighted:
                grouped_data = labelled_data.groupby([DATE]).apply(lambda x: np.average(x[RET_1], weights=x[MKTCAP]))
            else:
                grouped_data = labelled_data.groupby([DATE])[RET_1].mean()
            grouped_data = grouped_data.rename(label)
            grouped_data = _cumulate(grouped_data, cumulative)
            quantile_portfolio_returns = pd.concat([quantile_portfolio_returns, grouped_data], axis=1, sort=True)

        if show_plot:
            plt.figure()
            quantile_portfolio_returns.plot()
            if title:
                plt.title(title)
            else:
                plt.title(factor.upper())
            plt.ylabel("Return")
            plt.xlabel("Date")
            plt.legend(loc='upper left')
            plt.show()

        if show_bar_chart:
            plt.figure()
            quantile_result = portfolio.quantile_distribution_ratio(
                factor, chunk_num=chunk_num, cumulative=False, weighted=weighted, only_positive=only_positive,
                show_plot=False, show_bar_chart=False, title=None
            )
            quantile_result.mean(axis=0).plot(kind='bar')
            if title:
                plt.title(title)
            else:
                plt.title(factor.upper())
            plt.ylabel("Return")
            plt.xlabel("Group")
            plt.show()

        return quantile_portfolio_returns

    def rank_correlation(self, factor: str, ranked_by: str = RET_1, rolling: int = 6,
                         show_plot=False, title: str = '') -> DataFrame:
        portfolio = dc(self.dropna(subset=[ranked_by]))
        portfolio = portfolio.periodic_rank(factor=factor, drop_rank=False)
        factor_rank = "{factor}_rank".format(factor=factor)
        portfolio = portfolio.rename(index=str, columns={"rank": factor_rank})
        portfolio = portfolio.periodic_rank(factor=ranked_by, drop_rank=False)
        actual_rank = "{ranked_by}_rank".format(ranked_by=ranked_by)
        portfolio = portfolio.rename(index=str, columns={"rank": actual_rank})
        rank_ic = portfolio.groupby(by=[DATE]).apply(
            lambda x: 1 - (6 * ((x[factor_rank] - x[actual_rank]) ** 2).sum()) / (len(x) * (len(x) ** 2 - 1)))

        rank_ic = pd.DataFrame(rank_ic, columns=[RANK_CORRELATION])
        rolling_column_name = 'rolling_{}'.format(rolling)
        rank_ic[rolling_column_name] = rank_ic[RANK_CORRELATION].rolling(window=rolling).mean()

        if show_plot:
            rank_ic.plot()
            plt.title(title)
            plt.axhline(y=0, color='black')
            plt.ylabel('Rank IC')
            plt.xlabel('Date')
            plt.show()

        return rank_ic

    def show_plot(self, cumulative: bool = True, weighted: bool = False, title: str = None,
                  show_benchmark: bool = True, save: bool = False):
        portfolio = self.dropna(subset=[RET_1])

        if weighted:
            grouped_data = portfolio.groupby([DATE]).apply(lambda x: np.average(x[RET_1], weights=x[MKTCAP]))
        else:
            grouped_data = portfolio.groupby([DATE])[RET_1].mean()

        # noinspection PyProtectedMember
        grouped_data = _cumulate(grouped_data, cumulative)

        plt.figure()

        if show_benchmark:
            benchmark = self.get_benchmark().loc[:, [BENCHMARK_RET_1]]
            benchmark = _cumulate(benchmark, cumulative).dropna().reset_index(drop=False)
            grouped_data = grouped_data.reset_index(drop=False)
            grouped_data = pd.merge(grouped_data, benchmark, on=[DATE])
            grouped_data = grouped_data.rename(index=str, columns={
                RET_1: 'Portfolio',
                BENCHMARK_RET_1: self.benchmark
            })
            grouped_data = grouped_data.set_index(keys=[DATE])

        grouped_data.plot()

        if title:
            plt.title(title)
        else:
            plt.title("Portfolio Simulation")
        plt.ylabel("Return")
        plt.xlabel("Date")

        # save figure
        if save:
            plt.savefig(title)

        plt.show()


def _cumulate(ret, cumulative=True):
    ret.iloc[0] = 0
    if cumulative:
        ret = ret + 1
        ret = ret.cumprod()
        ret = ret - 1
    return ret


def _get_turnovers(portfolio: Portfolio, weight: str = None) -> Portfolio:
    all_companies = Portfolio(
        include_finance=True, include_holding=True, include_managed=True, include_suspended=True,
        start_date=portfolio[DATE].min(), end_date=portfolio[DATE].max()
    )[[DATE, CODE]]

    if weight:
        selected_portfolio = portfolio[[DATE, CODE, weight]]
    else:
        selected_portfolio = portfolio[[DATE, CODE]]
        selected_portfolio.loc[:, WEIGHT] = 1
        weight = WEIGHT

    weight_sum = selected_portfolio.groupby(DATE)[weight].sum()
    weight_sum.name = WEIGHT_SUM
    selected_portfolio = selected_portfolio.merge(weight_sum, on=DATE)
    selected_portfolio[weight] = selected_portfolio[weight] / selected_portfolio[WEIGHT_SUM]
    selected_portfolio = selected_portfolio[[DATE, CODE, weight]]
    selected_portfolio = selected_portfolio.merge(
        all_companies, how='outer', on=[DATE, CODE]).fillna(0).sort_values([DATE, CODE])
    selected_portfolio[TURNOVER] = selected_portfolio.groupby(CODE)[weight].apply(
        lambda x: x - x.shift(-1)
    )
    selected_portfolio = selected_portfolio.loc[selected_portfolio[TURNOVER] >= 0, :]
    selected_portfolio[TURNOVER] = selected_portfolio[TURNOVER].fillna(selected_portfolio[weight])
    turnovers = selected_portfolio.groupby(DATE)[TURNOVER].sum()

    return turnovers

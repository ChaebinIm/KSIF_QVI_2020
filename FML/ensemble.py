# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 11. 25.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ksif import Portfolio
from ksif.core.columns import *
from ksif.core.outcomes import *
from scipy.stats import spearmanr
from tqdm import tqdm

INTERSECTION = 'intersection'
GEOMETRIC = 'geometric'

QUANTILE = 'quantile'
PREDICTED_RET_1 = 'predict_return_1'
ACTUAL_RANK = 'actual_rank'
PREDICTED_RANK = 'predicted_rank'
COUNT = 'count'

RANK = 'rank'
CORRECT = 'correct'
RIGID_ACCURACY = 'rigid_accuracy'
DECILE_ACCURACY = 'decile_accuracy'
QUARTER_ACCURACY = 'quarter_accuracy'
HALF_ACCURACY = 'half_accuracy'

pf = Portfolio()
CD91_returns = pf.get_benchmark(CD91)[BENCHMARK_RET_1]
CD91_returns = CD91_returns.dropna()

actual_returns = pf[[DATE, CODE, RET_1]]


def get_intersection_ensemble_predictions(predictions, quantile: int = 40):
    """
    :return ensemble_predictions:
        DATE        | (datetime64)
        CODE        | (str)
    """
    selected_predictions = _select_predictions(predictions, quantile, [DATE, CODE])

    # Intersection
    ensemble_predictions = [selected_predictions[0]]
    for current_prediction in selected_predictions[1:]:
        previous_ensemble = ensemble_predictions[-1]
        current_ensemble = pd.merge(previous_ensemble, current_prediction, on=[DATE, CODE])
        ensemble_predictions.append(current_ensemble)
    for index, ensemble_prediction in enumerate(ensemble_predictions):
        ensemble_predictions[index] = pd.merge(
            ensemble_prediction, predictions[0].loc[:, [DATE, CODE]], on=[DATE, CODE]
        )
    return ensemble_predictions


def get_geometric_ensemble_predictions(predictions, quantile: int = 40):
    """
    :return ensemble_predictions:
        DATE        | (datetime64)
        CODE        | (str)
    """
    # Take exponential
    for prediction in predictions:
        prediction[PREDICTED_RET_1] = np.exp(prediction[PREDICTED_RET_1])

    # Geometric mean
    ensemble_predictions = [predictions[0]]
    for current_prediction in predictions[1:]:
        previous_ensemble = ensemble_predictions[-1]
        current_ensemble = current_prediction
        current_ensemble[PREDICTED_RET_1] = previous_ensemble[PREDICTED_RET_1] * current_prediction[PREDICTED_RET_1]
        ensemble_predictions.append(current_ensemble)
    for index, ensemble_prediction in enumerate(ensemble_predictions):
        ensemble_prediction[PREDICTED_RET_1] = ensemble_prediction[PREDICTED_RET_1] ** (1 / (index + 1))

    # Take log
    for ensemble_prediction in ensemble_predictions:
        ensemble_prediction[PREDICTED_RET_1] = np.log(ensemble_prediction[PREDICTED_RET_1])

    # Select the top quantile
    ensemble_predictions = _select_predictions(ensemble_predictions, quantile, [DATE, CODE])

    return ensemble_predictions


def _select_predictions(predictions, quantile, columns):
    selected_predictions = []
    for prediction in predictions:
        prediction.loc[:, RANK] = prediction.groupby(by=[DATE])[PREDICTED_RET_1].transform(
            lambda x: x.rank(ascending=False, pct=True)
        )
        selected_predictions.append(prediction.loc[prediction[RANK] <= (1 / quantile), columns])
    return selected_predictions


def _get_predictions(model_name, start_number, end_number):
    file_names = [
        '{}-{}.csv'.format(x, model_name) for x in range(start_number, end_number + 1)
    ]
    predictions = [
        pd.read_csv('prediction/{}/{}'.format(model_name, file_name), parse_dates=[DATE]) for file_name in file_names
    ]
    for prediction in predictions:
        prediction[ACTUAL_RANK] = prediction[[DATE, RET_1]].groupby(DATE).rank(ascending=False).reset_index(drop=True)
        prediction[PREDICTED_RANK] = prediction[[DATE, PREDICTED_RET_1]].groupby(DATE).rank(
            ascending=False).reset_index(drop=True)
    return predictions


METHODS = [
    INTERSECTION,
    GEOMETRIC
]

GET_ENSEMBLE_PREDICTIONS = {
    INTERSECTION: get_intersection_ensemble_predictions,
    GEOMETRIC: get_geometric_ensemble_predictions
}


def _cumulate(ret):
    from model import TRAIN_START_DATE
    """

    :param ret: (Series)
        key     DATE    | (datetime)
        column  RET_1   | (float)

    :return:
    """
    ret = pd.concat([ret, CD91_returns.loc[TRAIN_START_DATE:CD91_returns.index[-2]]], 1)
    ret = ret.iloc[:, 0].fillna(value=ret.iloc[:, 1])
    ret.iloc[0] = 0
    ret = ret + 1
    ret = ret.cumprod()
    ret = ret - 1
    return ret


def _get_file_name(method: str, model_name: str, quantile: int) -> str:
    result_file_name = '{}/{}-{}'.format(method.lower(), quantile, model_name)
    return result_file_name


def _calculate_accuracy(ensemble_portfolio, predictions, quantile):
    ensemble_portfolio[DATE] = pd.to_datetime(ensemble_portfolio[DATE])

    selected_predictions = _select_predictions(predictions[:1], quantile, [DATE, CODE])
    selected_prediction_count = selected_predictions[0].groupby(by=DATE).count()
    selected_prediction_count.rename(columns={CODE: COUNT}, inplace=True)
    selected_prediction_count.reset_index(drop=False, inplace=True)

    ensemble_portfolio_count = ensemble_portfolio[[DATE, CODE]].groupby(by=DATE).count()
    ensemble_portfolio_count.rename(columns={CODE: COUNT}, inplace=True)

    merged_portfolio = pd.merge(ensemble_portfolio, predictions[0][[DATE, CODE, ACTUAL_RANK]], on=[DATE, CODE])
    merged_portfolio = pd.merge(merged_portfolio, selected_prediction_count, on=DATE)
    merged_portfolio[CORRECT] = merged_portfolio[ACTUAL_RANK] <= merged_portfolio[COUNT]

    accuracies = merged_portfolio[[DATE, CORRECT]].groupby(DATE).sum()[CORRECT] / \
                 ensemble_portfolio_count[COUNT]
    accuracy = accuracies.mean()

    return accuracy


def get_ensemble(method: str, model_name: str, start_number: int = 0, end_number: int = 9, step: int = 1,
                 quantile: int = 40, show_plot=True):
    """

    :param method: (str)
    :param model_name: (str)
    :param start_number: (int)
    :param end_number: (int)
    :param step: (int)
    :param quantile: (int)
    :param show_plot: (bool)

    :return ensemble_summary: (DataFrame)
        PORTFOLIO_RETURN    | (float)
        ACTIVE_RETURN       | (float)
        ACTIVE_RISK         | (float)
        IR                  | (float)
        CAGR                | (float)
        RIGID_ACCURACY      | (float)
        DECILE_ACCURACY     | (float)
        QUARTER_ACCURACY    | (float)
        HALF_ACCURACY       | (float)
    :return ensemble_portfolios: ([Portfolio])
        DATE                | (datetime)
        CODE                | (str)
        RET_1               | (float)
    """
    # Check parameters
    assert method in METHODS, "method does not exist."
    assert end_number > start_number + 1, "end_number should be bigger than (start_number + 1)."
    assert step >= 1, "step should be a positive integer."
    assert quantile > 1, "quantile should be an integer bigger than 1."

    result_file_name = _get_file_name(method, model_name, quantile)

    predictions = _get_predictions(model_name, start_number, end_number)

    get_ensemble_predictions = GET_ENSEMBLE_PREDICTIONS[method]

    ensemble_predictions = get_ensemble_predictions(predictions, quantile)

    # Append actual returns
    ensemble_predictions = [pd.merge(ensemble_prediction, actual_returns, on=[DATE, CODE]) for
                            ensemble_prediction in ensemble_predictions]

    # Cumulative ensemble
    ensemble_numbers = pd.DataFrame(index=ensemble_predictions[0][DATE].unique())
    ensemble_cumulative_returns = pd.DataFrame(index=ensemble_predictions[0][DATE].unique())
    for index, ensemble_prediction in enumerate(ensemble_predictions):
        ensemble_number = ensemble_prediction.groupby(by=[DATE])[CODE].count()
        ensemble_return = ensemble_prediction.groupby(by=[DATE])[RET_1].mean()
        ensemble_cumulative_return = _cumulate(ensemble_return)

        if (index + 1) % step == 0:
            ensemble_numbers[index + 1] = ensemble_number
            ensemble_cumulative_returns[index + 1] = ensemble_cumulative_return

    # Fill nan
    ensemble_numbers.fillna(0, inplace=True)
    ensemble_cumulative_returns.fillna(method='ffill', inplace=True)
    ensemble_cumulative_returns.fillna(0, inplace=True)

    ensemble_portfolios = [Portfolio(ensemble_prediction) for ensemble_prediction in
                           ensemble_predictions[(step - 1)::step]]

    for ensemble_portfolio in ensemble_portfolios:
        if ensemble_portfolio.empty:
            return None, None

    ensemble_outcomes = [ensemble_portfolio.outcome() for ensemble_portfolio in ensemble_portfolios]
    portfolio_returns = [ensemble_outcome[PORTFOLIO_RETURN] for ensemble_outcome in ensemble_outcomes]
    active_returns = [ensemble_outcome[ACTIVE_RETURN] for ensemble_outcome in ensemble_outcomes]
    active_risks = [ensemble_outcome[ACTIVE_RISK] for ensemble_outcome in ensemble_outcomes]
    information_ratios = [ensemble_outcome[IR] for ensemble_outcome in ensemble_outcomes]
    sharpe_ratios = [ensemble_outcome[SR] for ensemble_outcome in ensemble_outcomes]
    MDDs = [ensemble_outcome[MDD] for ensemble_outcome in ensemble_outcomes]
    alphas = [ensemble_outcome[FAMA_FRENCH_ALPHA] for ensemble_outcome in ensemble_outcomes]
    betas = [ensemble_outcome[FAMA_FRENCH_BETA] for ensemble_outcome in ensemble_outcomes]
    CAGRs = [ensemble_outcome[CAGR] for ensemble_outcome in ensemble_outcomes]
    rigid_accuracies = [_calculate_accuracy(ensemble_portfolio, predictions, quantile) for
                        ensemble_portfolio in ensemble_portfolios]
    decile_accuracies = [_calculate_accuracy(ensemble_portfolio, predictions, 10) for
                         ensemble_portfolio in ensemble_portfolios]
    quarter_accuracies = [_calculate_accuracy(ensemble_portfolio, predictions, 4) for
                          ensemble_portfolio in ensemble_portfolios]
    half_accuracies = [_calculate_accuracy(ensemble_portfolio, predictions, 2) for
                       ensemble_portfolio in ensemble_portfolios]

    ensemble_summary = pd.DataFrame({
        PORTFOLIO_RETURN: portfolio_returns,
        ACTIVE_RETURN: active_returns,
        ACTIVE_RISK: active_risks,
        IR: information_ratios,
        SR: sharpe_ratios,
        MDD: MDDs,
        FAMA_FRENCH_ALPHA: alphas,
        FAMA_FRENCH_BETA: betas,
        CAGR: CAGRs,
        RIGID_ACCURACY: rigid_accuracies,
        DECILE_ACCURACY: decile_accuracies,
        QUARTER_ACCURACY: quarter_accuracies,
        HALF_ACCURACY: half_accuracies,
    }, index=ensemble_numbers.columns)
    ensemble_summary.to_csv('summary/' + result_file_name + '.csv')
    for ensemble_prediction in ensemble_predictions:
        ensemble_prediction[DATE] = pd.to_datetime(ensemble_prediction[DATE], format='%Y-%m-%d')

    # Plot
    if show_plot:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

        # Company number
        ensemble_numbers.plot(ax=axes[0], colormap='Blues')
        axes[0].set_title('{}:{}, Top {}-quantile'.format(method.title(), model_name, quantile))
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('# of companies')
        axes[0].legend(loc='upper left')

        # Cumulative return
        ensemble_cumulative_returns.plot(ax=axes[1], colormap='Blues')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Return')
        axes[1].legend(loc='upper left')

        # Information ratio
        # ensembles = ensemble_cumulative_returns.columns
        # trend_model = np.polyfit(ensembles, information_ratios, 1)
        # get_trend = np.poly1d(trend_model)
        # axes[2].plot(ensembles, information_ratios, 'black', ensembles, get_trend(ensembles), 'r--')
        # axes[2].set_ylim(0.3, 0.5)
        # axes[2].set_xlabel('# of ensembles')
        # axes[2].set_ylabel('Information ratio')

        plt.savefig('summary/' + result_file_name + '.png')
        fig.show()

    return ensemble_summary, ensemble_portfolios


# noinspection PyPep8Naming
def compare_ensemble(methods, models, quantiles, start_number: int = 0, end_number: int = 9, step: int = 1,
                     to_csv: bool = True, show_plot: bool = False):
    file_names = []
    CAGRs = []
    GAGR_rank_correlations = []
    CAGR_rank_p_values = []
    IRs = []
    IR_rank_correlations = []
    IR_rank_p_values = []
    SRs = []
    SR_rank_correlations = []
    SR_rank_p_values = []
    MDDs = []
    alphas = []
    alpha_rank_correlations = []
    alpha_rank_p_values = []
    betas = []
    rigid_accuracies = []
    decile_accuracies = []
    quarter_accuracies = []
    half_accuracies = []
    kospi_larges = []
    kospi_middles = []
    kospi_smalls = []
    kosdaq_larges = []
    kosdaq_middles = []
    kosdaq_smalls = []

    firms = Portfolio(include_holding=True, include_finance=True, include_managed=True, include_suspended=True).loc[:,
            [DATE, CODE, MKTCAP, EXCHANGE]]
    firms[DATE] = pd.to_datetime(firms[DATE])

    firms[RANK] = firms[[DATE, EXCHANGE, MKTCAP]].groupby([DATE, EXCHANGE]).rank(ascending=False)
    firms[KOSPI_LARGE] = firms.apply(
        lambda row: 1 if (row[EXCHANGE] == '유가증권시장') and (row[RANK] <= 100) else 0, axis=1)
    firms[KOSPI_MIDDLE] = firms.apply(
        lambda row: 1 if (row[EXCHANGE] == '유가증권시장') and (100 < row[RANK] <= 300) else 0, axis=1)
    firms[KOSPI_SMALL] = firms.apply(
        lambda row: 1 if (row[EXCHANGE] == '유가증권시장') and (300 < row[RANK]) else 0, axis=1)
    firms[KOSDAQ_LARGE] = firms.apply(
        lambda row: 1 if (row[EXCHANGE] == '코스닥') and (row[RANK] <= 100) else 0, axis=1)
    firms[KOSDAQ_MIDDLE] = firms.apply(
        lambda row: 1 if (row[EXCHANGE] == '코스닥') and (100 < row[RANK] <= 300) else 0, axis=1)
    firms[KOSDAQ_SMALL] = firms.apply(
        lambda row: 1 if (row[EXCHANGE] == '코스닥') and (300 < row[RANK]) else 0, axis=1)

    firms = firms.loc[
            :, [DATE, CODE, KOSPI_LARGE, KOSPI_MIDDLE, KOSPI_SMALL, KOSDAQ_LARGE, KOSDAQ_MIDDLE, KOSDAQ_SMALL]
            ]

    for method in methods:
        for quantile in quantiles:
            for model in tqdm(models):
                ensemble_summary, ensemble_portfolios = get_ensemble(
                    method, model_name=model, start_number=start_number, end_number=end_number, step=step,
                    quantile=quantile, show_plot=show_plot
                )

                if ensemble_summary is None and ensemble_portfolios is None:
                    continue

                ensemble_portfolio = pd.merge(ensemble_portfolios[-1], firms, on=[DATE, CODE])
                ensemble_portfolio_count = ensemble_portfolio[[DATE, CODE]].groupby(DATE).count()
                ensemble_portfolio_count.rename(columns={CODE: COUNT}, inplace=True)
                ensemble_portfolio_sum = ensemble_portfolio[[
                    DATE, KOSPI_LARGE, KOSPI_MIDDLE, KOSPI_SMALL, KOSDAQ_LARGE, KOSDAQ_MIDDLE, KOSDAQ_SMALL
                ]].groupby(DATE).sum()
                ensemble_portfolio_ratio = pd.merge(ensemble_portfolio_sum, ensemble_portfolio_count, on=DATE)
                ensemble_portfolio_ratio[KOSPI_LARGE] \
                    = ensemble_portfolio_ratio[KOSPI_LARGE] / ensemble_portfolio_ratio[COUNT]
                ensemble_portfolio_ratio[KOSPI_MIDDLE] \
                    = ensemble_portfolio_ratio[KOSPI_MIDDLE] / ensemble_portfolio_ratio[COUNT]
                ensemble_portfolio_ratio[KOSPI_SMALL] \
                    = ensemble_portfolio_ratio[KOSPI_SMALL] / ensemble_portfolio_ratio[COUNT]
                ensemble_portfolio_ratio[KOSDAQ_LARGE] \
                    = ensemble_portfolio_ratio[KOSDAQ_LARGE] / ensemble_portfolio_ratio[COUNT]
                ensemble_portfolio_ratio[KOSDAQ_MIDDLE] \
                    = ensemble_portfolio_ratio[KOSDAQ_MIDDLE] / ensemble_portfolio_ratio[COUNT]
                ensemble_portfolio_ratio[KOSDAQ_SMALL] \
                    = ensemble_portfolio_ratio[KOSDAQ_SMALL] / ensemble_portfolio_ratio[COUNT]

                file_names.append(_get_file_name(method, model, quantile))

                CAGRs.append(ensemble_summary[CAGR].values[-1])
                CAGR_rankIC = spearmanr(ensemble_summary[CAGR].values, ensemble_summary[CAGR].index)
                GAGR_rank_correlations.append(CAGR_rankIC[0])
                CAGR_rank_p_values.append(CAGR_rankIC[1])

                IRs.append(ensemble_summary[IR].values[-1])
                IR_rankIC = spearmanr(ensemble_summary[IR].values, ensemble_summary[IR].index)
                IR_rank_correlations.append(IR_rankIC[0])
                IR_rank_p_values.append(IR_rankIC[1])

                SRs.append(ensemble_summary[SR].values[-1])
                SR_rankIC = spearmanr(ensemble_summary[SR].values, ensemble_summary[SR].index)
                SR_rank_correlations.append(SR_rankIC[0])
                SR_rank_p_values.append(SR_rankIC[1])

                MDDs.append(ensemble_summary[MDD].values[-1])

                alphas.append(ensemble_summary[FAMA_FRENCH_ALPHA].values[-1])
                alpha_rankIC = spearmanr(ensemble_summary[FAMA_FRENCH_ALPHA].values,
                                         ensemble_summary[FAMA_FRENCH_ALPHA].index)
                alpha_rank_correlations.append(alpha_rankIC[0])
                alpha_rank_p_values.append(alpha_rankIC[1])
                betas.append(ensemble_summary[FAMA_FRENCH_BETA].values[-1])

                rigid_accuracies.append(ensemble_summary[RIGID_ACCURACY].values[-1])
                decile_accuracies.append(ensemble_summary[DECILE_ACCURACY].values[-1])
                quarter_accuracies.append(ensemble_summary[QUARTER_ACCURACY].values[-1])
                half_accuracies.append(ensemble_summary[HALF_ACCURACY].values[-1])

                kospi_larges.append(ensemble_portfolio_ratio[KOSPI_LARGE].mean())
                kospi_middles.append(ensemble_portfolio_ratio[KOSPI_MIDDLE].mean())
                kospi_smalls.append(ensemble_portfolio_ratio[KOSPI_SMALL].mean())
                kosdaq_larges.append(ensemble_portfolio_ratio[KOSDAQ_LARGE].mean())
                kosdaq_middles.append(ensemble_portfolio_ratio[KOSDAQ_MIDDLE].mean())
                kosdaq_smalls.append(ensemble_portfolio_ratio[KOSDAQ_SMALL].mean())

    comparison_result = pd.DataFrame(data={
        'Model': file_names,
        'CAGR': CAGRs,
        'CAGR RC': GAGR_rank_correlations,
        'CAGR RC p-value': CAGR_rank_p_values,
        'IR': IRs,
        'IR RC': IR_rank_correlations,
        'IR RC p-value': IR_rank_p_values,
        'SR': SRs,
        'SR RC': SR_rank_correlations,
        'SR RC p-value': SR_rank_p_values,
        'FF alpha': alphas,
        'FF alpha RC': alpha_rank_correlations,
        'FF alpha RC p-value': alpha_rank_p_values,
        'FF betas': betas,
        'MDD': MDDs,
        'Rigid accuracy': rigid_accuracies,
        'Decile accuracy': decile_accuracies,
        'Quarter accuracy': quarter_accuracies,
        'Half accuracy': half_accuracies,
        'KOSPI Large': kospi_larges,
        'KOSPI Middle': kospi_middles,
        'KOSPI Small': kospi_smalls,
        'KOSDAQ Large': kosdaq_larges,
        'KOSDAQ Middle': kosdaq_middles,
        'KOSDAQ Small': kosdaq_smalls,
    })

    if to_csv:
        comparison_result.to_csv('summary/comparison_result.csv', index=False)

    return comparison_result


if __name__ == '__main__':
    models = [
        'DNN8_2-sector-linear-he_uniform-glorot_uniform-none'
    ]
    methods = [
        # INTERSECTION,
        GEOMETRIC
    ]
    quantiles = [
        2,
        5,
        10,
        20,
        40
    ]
    compare_ensemble(methods, models, quantiles, start_number=0, end_number=9, step=1, to_csv=True, show_plot=True)

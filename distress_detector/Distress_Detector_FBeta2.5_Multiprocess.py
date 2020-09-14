import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
from datetime import datetime
from dateutil.relativedelta import *
from tqdm import tqdm
from multiprocessing import Pool
from time import sleep
import warnings
warnings.filterwarnings(action='ignore')


def distress_detector(target_date):
    # input example('Data/msf_with_index_ret.pck', False, backtest_start_date = '2011-01-01',
    #               backtest_end_date = '2018-01,01',exch_type = 'KOSDAQ', target_date = '2018-01-01')

    # Parameter_Setting
    data_path = './data/2020-07-31_company.pck'
    backtest_flag = False
    parameter_weight_recall = 2.5
    if backtest_flag == True and (backtest_start_date == None or backtest_end_date == None
                                  or exch_type == None):
        print('Please include exch_type, backtest_start_date,and backtest_end_date')
        return
    elif backtest_flag == False and (target_date == None):
        print('Please include target_date')
        return

    # Read data based on file types
    def read_file():
        temp = data_path[-3:]
        if temp.lower() == 'csv':
            data = pd.read_csv(data_path, parse_dates=['mdate'])
        elif temp.lower() == 'pck':
            data = pd.read_pickle(data_path)
        else:
            print('Unsupported Data Type')
        return data

    # Factor Preparation with Date Slicing
    def factor_prep(data):
        data = data.set_index('mdate')
        data.index = pd.to_datetime(data.index)

        # Sorting by Exchange
        data['index_ret'] = 0
        if exch_type.upper() == 'KSE' or exch_type.upper() == 'KOSPI':
            data = data[data['exchcd'] == 'KSE']
            data['index_ret'] = data.loc[:, 'vwret_kospi']
        elif exch_type.upper() == 'KOSDAQ':
            data = data[data['exchcd'] == 'KOSDAQ']
            data['index_ret'] = data.loc[:, 'vwret_kosdaq']
        else:
            print('Error in Exchange Name')

        # Adding log return values to data
        data['log_return'] = np.log(1 + data['ret'])

        # Adding sum of me per exchange type
        data['sum_of_me_each_exchcd'] = data.groupby(['exchcd', 'mdate'])['me'].transform('sum')

        # "Trading Halt Reason Filter => trading_halt_LEAD_1
        trading_halt_reason_key = keyword = ['상장폐지', '불성실', '회생', '실질심사', '파산', '상장적격',
                                             '자본잠식', '감사의견거절', '관리종목', '미제출']
        data['trading_halt_during_month_LEAD_1'] = 0
        temp = data.groupby('code').shift(-1)['trading_halt_reason']
        temp = temp.str.contains(pat='|'.join(trading_halt_reason_key), regex=True, na=False)
        data.loc[temp, 'trading_halt_during_month_LEAD_1'] = 1

        # Consolidated Factor for managed, halted, delisted   ==>   DSTR
        cond_1 = data['managed_during_month_LEAD_1'] == 0
        cond_2 = data['trading_halt_during_month_LEAD_1'] == 1
        cond_3 = data['delisted_LEAD_1'] == 1
        data['DSTR'] = 0
        DSTR = data['DSTR']
        DSTR[cond_1 | cond_2 | cond_3] = 1  # if distressed DSTR = 1 otherwise 0

        # adjusting indicators
        me = data['me'] * 1000000
        ltq = data['ltq'] * 1000
        niq = data['niq'] * 1000
        atq = data['atq'] * 1000
        sum_of_me_each_exchcd = data['sum_of_me_each_exchcd'] * 1000000
        prccadj = data['prccadj_hypo']
        log_return = data['log_return']
        vol_3m = data['vol_3m']
        index_ret = data['index_ret']

        # Adding columns with new factors
        data['NIMTA'] = niq / (me + ltq)
        data['TLMTA'] = ltq / (me + ltq)
        data['EXRET'] = log_return - index_ret
        data['RSIZE'] = np.log(me / sum_of_me_each_exchcd)
        data['SIGMA'] = vol_3m
        data['MB'] = me / (atq - ltq)
        data['LPRICE'] = np.log(prccadj)

        # Drop Infinite or NA values in Data
        temp = ['NIMTA', 'TLMTA', 'EXRET', 'RSIZE', 'SIGMA', 'MB', 'LPRICE']
        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=temp)

        return data

    # Sampling with conditions
    def sample_wcond(data):

        # TDSTR and FDSTR setting
        global TDSTR
        TDSTR = data[data['DSTR'] == 1]
        TDSTR = TDSTR.sort_index()
        TDSTR.index = pd.to_datetime(TDSTR.index)
        FDSTR = data.loc[~data['code'].isin(TDSTR['code'])]
        FDSTR = FDSTR.sort_index()
        FDSTR.index = pd.to_datetime(FDSTR.index)

        # median number of overlapped distressed firms
        num_dup = int(TDSTR[TDSTR.duplicated('code', keep=False)].groupby('code').size().median())

        # Sampling Function
        def sample(TDSTR, FDSTR):
            func = lambda x: pd.DataFrame.sample(x, n=num_dup, random_state=seed_num)
            FDSTR['mdate'] = FDSTR.index
            a = FDSTR.groupby('code').apply(func)
            result = a.set_index('mdate').sort_index()
            result.index = pd.to_datetime(result.index)
            return result

        # firm that are overlapped less than num_dup times
        temp = FDSTR[FDSTR.duplicated('code', keep=False)]['code'].value_counts() < num_dup
        temp1 = FDSTR[FDSTR['code'].isin(temp[temp].index)]
        # unique firms
        temp2 = FDSTR[~FDSTR.duplicated('code', keep=False)]
        # firms that are overlapped equal or more than num_dup times
        temp = FDSTR[FDSTR.duplicated('code', keep=False)]['code'].value_counts() >= num_dup
        temp3 = sample(TDSTR, FDSTR[FDSTR['code'].isin(temp[temp].index)])
        temp = pd.concat([temp1, temp2])
        temp = pd.concat([temp, temp3])
        return temp

    # Dynamic Logistic Regression and Bagging
    def dynamic_logistic(data, prediction_data, num_iter, solver_type, scaler_type):

        from sklearn.linear_model import LogisticRegression
        import pickle
        if scaler_type.lower() == 'minmaxscaler':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif scaler_type.lower() == 'standardscaler':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif scaler_type.lower() == 'maxabsscaler':
            from sklearn.preprocessing import MaxAbsScaler
            scaler = MaxAbsScaler()
        elif scaler_type.lower() == 'robustscaler':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()

        # Iterations for Bagging Process Begin Here
        model = {}
        for model_num in range(num_iter):
            # Set seed number
            global seed_num
            seed_num = model_num + 1

            # Formation of complete train data
            temp = sample_wcond(data)
            train_set = pd.concat([TDSTR, temp])

            # Model Fitting
            # Standardization with StandardScaler
            factors = ['NIMTA', 'TLMTA', 'EXRET', 'RSIZE', 'SIGMA', 'MB', 'LPRICE']
            temp1 = scaler.fit_transform(train_set.loc[:, factors])
            temp2 = train_set['DSTR']
            clf = LogisticRegression(solver=solver_type)
            clf.fit(temp1, temp2)
            model[model_num] = pickle.dumps(clf)

        # Define prediction_data
        data = prediction_data

        # Dropping infinite and NA values and data Standardization
        temp = ['code', 'firmname', 'NIMTA', 'TLMTA', 'EXRET', 'RSIZE', 'SIGMA', 'MB', 'LPRICE', 'DSTR']
        valid_set = data.replace([np.inf, -np.inf], np.nan).dropna(subset=temp)
        valid_train = scaler.fit_transform(valid_set.loc[:, factors])

        # Iterating Prediction of DSTR Probability
        temp = []
        for model_num in range(num_iter):
            clf = pickle.loads(model[model_num])
            y_pred = clf.predict_proba(valid_train)
            temp.append(f'pred_{model_num + 1}')
            valid_set[f'pred_{model_num + 1}'] = y_pred[:, 1]
            valid_set['predict'] = valid_set[temp].mean(axis=1)

        valid_set['REAL'] = valid_set['DSTR']

        # Return values needed
        temp = ['code', 'firmname', 'NIMTA', 'TLMTA', 'EXRET', 'RSIZE', 'SIGMA', 'MB', 'LPRICE', 'DSTR', 'predict',
                'REAL']

        return valid_set[temp]

    # Validate_cutoff
    def validate_cutoff(valid_data):

        def filter_by_cutoff(value_cut, df):
            col_name = str(value_cut)
            df[col_name] = 0
            df[col_name] = df['predict'].apply(lambda x: 1 if x > value_cut else 0)

        def summary_score(need_data):

            REAL = need_data['REAL']
            TARGET = need_data.iloc[:, (list(need_data.columns).index('REAL') + 1):]
            predict_summary = pd.DataFrame(columns=['precision_score', 'recall_score', 'f1_score'])

            for i in TARGET.columns:
                p_s = precision_score(REAL, TARGET[i])
                r_s = recall_score(REAL, TARGET[i])
                f_s = ((1 + parameter_weight_recall ** 2) * (p_s * r_s)) / ((parameter_weight_recall ** 2) * p_s + r_s)
                temp_f = pd.DataFrame([[p_s, r_s, f_s]], columns=['precision_score', 'recall_score', 'f1_score'],
                                      index=[i])

                predict_summary = predict_summary.append(temp_f)
            return (predict_summary)

        iter_num = [i / 100 for i in range(30, 90, 1)]
        for i in iter_num:
            filter_by_cutoff(i, valid_data)

        index_all = list(pd.Series(valid_data.index).apply(lambda x: datetime.strftime(x, "%Y-%m")).unique())
        all_valid_summmary = pd.DataFrame(columns=['precision_score', 'recall_score', 'f1_score', 'date_y_m'])

        for i in index_all:
            temp_valid_data = valid_data[i]
            temp_valid_summary = summary_score(temp_valid_data)
            date_y_m_value = i
            temp_valid_summary['date_y_m'] = date_y_m_value
            all_valid_summmary = all_valid_summmary.append(temp_valid_summary)

        all_valid_summmary['cutoff'] = all_valid_summmary.index
        all_valid_summmary = all_valid_summmary.reset_index(drop=True)

        return (all_valid_summmary)

    def final_summary(test_result):

        mdate = list(test_result.index.unique().strftime("%Y-%m"))[0]
        REAL = test_result['REAL']
        predict = test_result['predict_final']
        p_s = precision_score(REAL, predict)
        r_s = recall_score(REAL, predict)
        f_s = f1_score(REAL, predict)

        distress_num = test_result['REAL'].sum()
        predict_num = test_result['predict_final'].sum()
        REAL_AND_predict_num = sum((test_result['REAL'] + test_result['predict_final']) == 2)
        final_result = pd.DataFrame([[mdate, distress_num, predict_num, REAL_AND_predict_num, p_s, r_s, f_s]],
                                    columns=['mdate', 'distress_num', 'predict_num', 'REAL_AND_predict_num',
                                             'precision_score', 'recall_score', 'f1_score'])
        return (final_result)

    # Grid Search parameter setting
    from itertools import product
    parameter_set = [[5, 10, 20, 35, 50], ['saga', 'sag'],
                     ['standardscaler', 'robustscaler']]

    parameter_set_frame = pd.DataFrame(list(product(*parameter_set)), columns=['num_iter', 'solver_type',
                                                                               'scaler_type'])

    # Final Output Columns
    result_final_all = pd.DataFrame(columns=['mdate', 'distress_num', 'predict_num', 'REAL_AND_predict_num',
                                             'precision_score', 'recall_score', 'f1_score', 'cutoff',
                                             'num_iter', 'solver_type', 'scaler_type'])

    # Data loading & prepping
    # Set exch_type_list for actual test
    if backtest_flag == False:
        exch_type_list = ['KOSPI', 'KOSDAQ']
    else:
        exch_type_list = [exch_type]
    result_dict = {}

    for exch_name in exch_type_list:
        print(f'Processing {target_date} {exch_name.upper()}')
        exch_type = exch_name
        temp = read_file()
        data_all = factor_prep(temp)
        if backtest_flag == True:
            train_begin = backtest_start_date
            end_date = datetime.strftime(datetime.strptime(backtest_end_date, '%Y-%m-%d'), '%Y-%m')
        elif backtest_flag == False:
            train_begin = datetime.strftime(datetime.strptime(target_date, '%Y-%m-%d') - relativedelta(months=73) + relativedelta(day=1), '%Y-%m-%d')
            end_date = datetime.strftime(datetime.strptime(target_date, '%Y-%m-%d'), '%Y-%m')
            # Rolling windows by shifting date

        while True:
            train_end = datetime.strptime(train_begin, '%Y-%m-%d') + relativedelta(years=+5) + relativedelta(day=31)
            valid_begin = train_end + relativedelta(months=1) + relativedelta(day=1)
            valid_end = valid_begin + relativedelta(years=+1) - relativedelta(months=+1) + relativedelta(day=31)
            test_date = valid_end + relativedelta(months=+1) + relativedelta(day=31)
            test_date = datetime.strftime(test_date, '%Y-%m')

            train_data = data_all.loc[train_begin: datetime.strftime(train_end, '%Y-%m-%d')]
            valid_data = data_all.loc[
                         datetime.strftime(valid_begin, '%Y-%m-%d'): datetime.strftime(valid_end, '%Y-%m-%d')]
            test_data = data_all.loc[test_date]

            temp_data_result = pd.DataFrame(columns=['precision_score', 'recall_score', 'f1_score',
                                                     'date_y_m', 'num_iter', 'solver_type', 'scaler_type'])

            # Finding best parameters in validation set
            for i in tqdm(range(len(parameter_set_frame.index)),
                                   desc=f'Test_Date: {test_date}'):
                valid_predict = dynamic_logistic(train_data, valid_data,
                                                 parameter_set_frame['num_iter'][i],
                                                 parameter_set_frame['solver_type'][i],
                                                 parameter_set_frame['scaler_type'][i])

                valid_predict_result = validate_cutoff(valid_predict)
                valid_predict_result['num_iter'] = parameter_set_frame['num_iter'][i]
                valid_predict_result['solver_type'] = parameter_set_frame['solver_type'][i]
                valid_predict_result['scaler_type'] = parameter_set_frame['scaler_type'][i]
                temp_data_result = temp_data_result.append(valid_predict_result)
                sleep(0.1)

            best_valid_parameter = pd.DataFrame(
                temp_data_result.groupby(['cutoff', 'num_iter', 'solver_type', 'scaler_type']).mean()[
                    'f1_score'].idxmax()).T
            best_valid_parameter.columns = ['cutoff', 'num_iter', 'solver_type', 'scaler_type']
            test_result = dynamic_logistic(train_data, test_data,
                                           best_valid_parameter['num_iter'][0],
                                           best_valid_parameter['solver_type'][0],
                                           best_valid_parameter['scaler_type'][0])

            # Adding Predict_final column and Setting Cutoff parameter with "best_valid_parameter"
            test_result['predict_final'] = 0
            test_result['predict_final'] = test_result['predict'].apply(lambda x: 1
            if x > float(best_valid_parameter['cutoff']) else 0)

            if backtest_flag == True:

                test_result_summary = final_summary(test_result)
                test_result_summary['cutoff'] = float(best_valid_parameter['cutoff'])
                test_result_summary['num_iter'] = best_valid_parameter['num_iter']
                test_result_summary['solver_type'] = best_valid_parameter['solver_type']
                test_result_summary['scaler_type'] = best_valid_parameter['scaler_type']

                test_result_summary['train_begin'] = train_begin
                test_result_summary['train_end'] = train_end
                test_result_summary['valid_begin'] = valid_begin
                test_result_summary['valid_end'] = valid_end
                test_result_summary['test_date'] = test_date
                test_result_summary['num_total_firms'] = len(test_data)

                result_final_all = result_final_all.append(test_result_summary)
                result_final_all = result_final_all[
                    ['test_date', 'train_begin', 'train_end', 'valid_begin', 'valid_end', 'num_total_firms',
                     'distress_num', 'predict_num', 'REAL_AND_predict_num',
                     'precision_score',
                     'recall_score', 'f1_score', 'cutoff', 'num_iter', 'solver_type',
                     'scaler_type']]

            elif backtest_flag == False:
                result_final_all = test_result[['code', 'predict', 'predict_final']]
                break

            if test_date == end_date:
                break

            train_begin = datetime.strptime(train_begin, '%Y-%m-%d') + relativedelta(months=+1)
            train_begin = datetime.strftime(train_begin, '%Y-%m-%d')

        result_dict[exch_name] = result_final_all

    if backtest_flag == False:
        final_result = pd.concat([result_dict['KOSPI'], result_dict['KOSDAQ']], axis=0)
        final_result.columns = ['code', 'distress_score', 'distress_indicator']
    else:
        final_result = result_dict[exch_name]
    return final_result

def parallelized_distress_detector(target_dates, number_of_cores):
    if __name__ == '__main__':
        pool = Pool(number_of_cores)
        temp_result = pd.concat(pool.map(distress_detector, target_dates), axis=0)
        data = pd.read_pickle('./data/2020-07-31_company.pck')
        final_result = pd.merge(left=data, right=temp_result, how='left', on=['code', 'mdate'], sort=False)
        final_result.to_pickle('./Distress_Detector_2.5_from{}to{}.pck'.format(start_date, end_date))
        pool.close()
        pool.join()

#Run Model with Multiprocessing
start_date = '2016-01-01'
end_date = '2020-07-31'
target_dates = [datetime.strftime(date,'%Y-%m-%d') for date in pd.date_range(start_date, end_date, freq="M")]
parallelized_distress_detector(target_dates, 20)
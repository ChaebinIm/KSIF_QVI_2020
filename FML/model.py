# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-23
"""
import gc
from datetime import datetime
from pathlib import Path

import keras
import tensorflow as tf
from arch import arch_model
from dateutil.relativedelta import relativedelta
from keras import backend as k
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential

from ensemble import GET_ENSEMBLE_PREDICTIONS, PREDICTED_RET_1
from settings import *

TRAINING_MONTHS = 36  # After 36 months training, test 1 month.

TRAIN_START_DATE = (
        datetime.strptime(START_DATE, '%Y-%m-%d') + relativedelta(months=TRAINING_MONTHS + 1)
).strftime('%Y-%m-%d')

pf = Portfolio()
pf = pf[pf[DATE] >= START_DATE]
months = sorted(pf[DATE].unique())[:-1]

result_columns = [RET_1]


def get_train_test_set(data_set_key, test_month):
    training_set = get_data_set(data_set_key)
    test_set = get_data_set(data_set_key)

    if test_month in months:
        test_index = months.index(test_month)
    else:
        test_index = len(months)
    assert test_index - TRAINING_MONTHS - 1 >= 0, "test_month is too early"

    train_start_month = months[test_index - TRAINING_MONTHS - 1]

    training_set = training_set.loc[(training_set[DATE] >= train_start_month) & (training_set[DATE] < test_month), :]
    test_set = test_set.loc[test_set[DATE] == test_month, :]

    return training_set, test_set


def train_model(month, param, early_stop, batch_normalization):
    data_trains, data_test = get_train_test_set(data_set_key=param[DATA_SET], test_month=month)

    data_train_array = data_trains.values
    data_test_array = data_test.values

    x_train = data_train_array[:, 3:]
    y_train = data_train_array[:, 2:3]
    x_test = data_test_array[:, 3:]
    actual_test = data_test.loc[:, [DATE, CODE, RET_1]].reset_index(drop=True)

    input_dim = x_train.shape[1]

    # Parameters
    batch_size = param[BATCH_SIZE]
    epochs = param[EPOCHS]
    activation = get_activation(param[ACTIVATION])
    bias_initializer = get_initializer(param[BIAS_INITIALIZER])
    kernel_initializer = get_initializer(param[KERNEL_INITIALIZER])
    bias_regularizer = get_regularizer(param[BIAS_REGULARIZER])
    hidden_layer = get_hidden_layer(param[HIDDEN_LAYER])
    dropout = param[DROPOUT]
    dropout_rate = param[DROPOUT_RATE]

    model = Sequential()
    model.add(Dense(hidden_layer[0], input_dim=input_dim,
                    activation=activation,
                    bias_initializer=bias_initializer,
                    kernel_initializer=kernel_initializer,
                    bias_regularizer=bias_regularizer
                    ))
    if batch_normalization:
        model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(dropout_rate))

    for hidden_layer in hidden_layer[1:]:
        model.add(Dense(hidden_layer,
                        activation=activation,
                        bias_initializer=bias_initializer,
                        kernel_initializer=kernel_initializer
                        ))
        if batch_normalization:
            model.add(BatchNormalization())
        if dropout:
            model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    model.compile(loss=keras.losses.mse,
                  optimizer=keras.optimizers.Adam())

    if early_stop:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0,
                  callbacks=[EarlyStopping(patience=10)],
                  validation_split=0.2)
    else:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0)

    return model, x_test, actual_test


def get_file_name(param) -> str:
    file_name = '{hidden_layer}-{data_set}-{activation}-{bias_initializer}-{kernel_initializer}-{bias_regularizer}'.format(
        hidden_layer=param[HIDDEN_LAYER],
        data_set=param[DATA_SET],
        activation=param[ACTIVATION],
        bias_initializer=param[BIAS_INITIALIZER],
        kernel_initializer=param[KERNEL_INITIALIZER],
        bias_regularizer=param[BIAS_REGULARIZER],
    )
    if param[DROPOUT]:
        file_name = file_name + '-{}'.format(param[DROPOUT_RATE])

    return file_name


def get_predictions(model, x_test, actual_y=None):
    predict_ret_1 = 'predict_' + RET_1
    actual_rank = 'actual_rank'
    predicted_rank = 'predicted_rank'

    prediction = model.predict(x_test, verbose=0)
    if isinstance(actual_y, pd.DataFrame):
        df_prediction = pd.concat(
            [actual_y,
             pd.DataFrame(prediction, columns=[predict_ret_1])],
            axis=1)
        df_prediction['diff'] = df_prediction[RET_1] - df_prediction[predict_ret_1]
        df_prediction[actual_rank] = df_prediction[RET_1].rank(ascending=False)
        df_prediction[predicted_rank] = df_prediction[predict_ret_1].rank(ascending=False)
    else:
        df_prediction = pd.DataFrame(prediction, columns=[predict_ret_1])

    return df_prediction


def backtest(param, start_number=0, end_number=9, max_pool=os.cpu_count()):
    print("Param: {}".format(param))
    pool_num = min(max_pool, end_number - start_number + 1)
    print("From {} to {} with {} processes.".format(start_number, end_number, pool_num))

    test_pf = pf.loc[pf[DATE] >= TRAIN_START_DATE, :]
    test_months = sorted(test_pf[DATE].unique())[:-1]

    with Pool(pool_num) as p:
        # for case_number in range(start_number, end_number + 1):
        #     p.apply_async(_backtest, args=(case_number, param, test_months))
        results = [p.apply_async(_backtest, (case_number, param, test_months))
                   for case_number in range(start_number, end_number + 1)]
        for r in results:
            r.wait()
        [result.get() for result in results]
        p.close()
        p.join()


def _backtest(case_number: int, param: dict, test_months: list, x_test_scaling=False, y_test_scaling=False,
              control_volatility_regime=False, early_stop=True, batch_normalization=True):
    tf.logging.set_verbosity(3)
    # TensorFlow wizardry
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Create a session with the above options specified.
    k.set_session(tf.Session(config=config))

    file_name = get_file_name(param)
    desc = "#{0:2d}".format(case_number)
    df_predictions = pd.DataFrame()

    for month in tqdm(test_months, desc=desc):
        if control_volatility_regime:
            # Calculate past actual volatilities
            bm = pf.get_benchmark(KOSPI)
            returns = bm.loc[:, BENCHMARK_RET_1] * 100
            returns = returns.dropna()
            window = 10
            actual_vol = returns.rolling(window).var()
            # Predict a future volatility
            ret_rolling = returns.loc[returns.index < month]
            am = arch_model(ret_rolling, vol='Garch', p=1, o=0, q=1, dist='Normal')
            res = am.fit(update_freq=0, disp='off')
            vol = res.forecast(horizon=1).variance.dropna()
            vol = vol.iloc[0, 0]
            execution = vol < actual_vol.loc[returns.index < month].quantile(.85)

            # Determine whether invest or not
            # If determined investing, train a model and get predictions.
            # Or skip this month.
            if not execution:
                continue

        model, x_test, y_test = train_model(month, param, early_stop=early_stop,
                                            batch_normalization=batch_normalization)

        # MinMaxScaling x_test
        if x_test_scaling:
            minmaxscaling = lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
            for i, j in enumerate(x_test):
                x_test[i] = minmaxscaling(j)

        # MinMaxScaling y_test
        if y_test_scaling:
            y_test[RET_1] = (y_test[RET_1] - y_test[RET_1].min()) / (y_test[RET_1].max() - y_test[RET_1].min())

        df_prediction = get_predictions(model, x_test, y_test)
        df_predictions = pd.concat([df_predictions, df_prediction], axis=0, ignore_index=True)
        gc.collect()

    # If a directory for this model does not exist, make it.
    data_dir = 'prediction/{}'.format(file_name)
    if not Path(data_dir).exists():
        os.makedirs(data_dir)

    # Save the result of the model with a case number.
    df_predictions.to_csv(
        '{data_dir}/{case_number}-{file_name}.csv'.format(
            data_dir=data_dir,
            case_number=case_number,
            file_name=file_name
        ),
        index=False
    )

    # Clean up the memory
    k.get_session().close()
    k.clear_session()
    tf.reset_default_graph()


# noinspection PyUnresolvedReferences
def get_forward_predict(param, quantile, model_num, method, x_test_scaling=False):
    print("Param: {}".format(param))

    recent_data_set = param[DATA_SET] + '_recent'
    x_test = pd.read_hdf('data/{}.h5'.format(recent_data_set))

    month = x_test[DATE].iloc[0]
    codes = x_test[[CODE]]
    x_test = x_test.drop([DATE, CODE], axis=1)

    # MinMaxScaling x_test
    if x_test_scaling:
        minmaxscaling = lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
        for i, j in enumerate(x_test):
            x_test[i] = minmaxscaling(j)

    with Pool(min(os.cpu_count(), model_num)) as p:
        # noinspection PyTypeChecker
        results = [p.apply_async(_get_forward_predict, t) for t in zip(
            [codes for _ in range(model_num)],
            [month for _ in range(model_num)],
            [param for _ in range(model_num)],
            [x_test for _ in range(model_num)],
        )]
        for r in results:
            r.wait()
        results = [result.get() for result in results]
        p.close()
        p.join()

        # 0 = intersection / 1 = geometric
        get_ensemble_predictions = GET_ENSEMBLE_PREDICTIONS[method]
        ensemble_predictions = get_ensemble_predictions(results, quantile)

        ensemble_predictions = ensemble_predictions[-1][CODE]

        # Save predictions
        ensemble_predictions.to_csv('forward_predict/forward_predictions.csv', index=False)


def _get_forward_predict(codes, month, param, x_test, early_stop=True, batch_normalization=True):
    tf.logging.set_verbosity(3)
    # TensorFlow wizardry
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Create a session with the above options specified.
    k.set_session(tf.Session(config=config))

    model, _, _ = train_model(month, param, early_stop=early_stop, batch_normalization=batch_normalization)
    # get forward prediction
    forward_predictions = get_predictions(model, x_test)
    codes[PREDICTED_RET_1] = forward_predictions
    df_forward_predictions = codes
    df_forward_predictions[DATE] = month

    # Clean up the memory
    k.get_session().close()
    k.clear_session()
    tf.reset_default_graph()

    return df_forward_predictions

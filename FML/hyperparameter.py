import tensorflow as tf
import keras
from keras import backend as k
import talos as ta
from keras.layers import Dropout, Dense, BatchNormalization
from keras.models import Sequential
from talos.model.early_stopper import early_stopper

from settings import *


def get_train_test_set(data_set_key, test_month):
    training_set = get_data_set(data_set_key)
    test_set = get_data_set(data_set_key)

    train_start_month = START_DATE

    training_set = training_set.loc[(training_set[DATE] >= train_start_month) & (training_set[DATE] < test_month), :]
    test_set = test_set.loc[test_set[DATE] == test_month, :]

    return training_set, test_set


def fml_model(x_train, y_train, x_val, y_val, params):
    input_dim = x_train.shape[1]

    # Parameters
    batch_size = params[BATCH_SIZE]
    epochs = params[EPOCHS]
    activation = params[ACTIVATION]
    bias_initializer = params[BIAS_INITIALIZER]
    kernel_initializer = params[KERNEL_INITIALIZER]
    bias_regularizer = params[BIAS_REGULARIZER]
    hidden_layer = params[HIDDEN_LAYER]
    dropout_rate = params[DROPOUT_RATE]

    model = Sequential()
    model.add(Dense(hidden_layer[0], input_dim=input_dim,
                    activation=activation,
                    bias_initializer=bias_initializer,
                    kernel_initializer=kernel_initializer,
                    bias_regularizer=bias_regularizer
                    ))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    for hidden_layer in hidden_layer[1:]:
        model.add(Dense(hidden_layer,
                        activation=activation,
                        bias_initializer=bias_initializer,
                        kernel_initializer=kernel_initializer
                        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    model.compile(loss=keras.losses.mse,
                  optimizer=keras.optimizers.Adam())
    out = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_data=[x_val, y_val],
                    callbacks=[early_stopper(epochs, mode='strict')])

    return out, model


if __name__ == '__main__':

    tf.logging.set_verbosity(3)
    # TensorFlow wizardry
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Create a session with the above options specified.
    k.set_session(tf.Session(config=config))

    month = '2018-11-30'

    data_train, data_test = get_train_test_set(data_set_key=ALL, test_month=month)

    # Make data a numpy array
    data_train_array = data_train.values
    data_test_array = data_test.values

    X_train = data_train_array[:, 3:]
    y_train = data_train_array[:, 2:3]
    X_test = data_test_array[:, 3:]
    y_text = data_test_array[:, 2:3]

    p = {
        BATCH_SIZE: [300],
        EPOCHS: [100],
        ACTIVATION: [linear, relu],
        BIAS_INITIALIZER: [
            lecun_uniform(), he_uniform(), glorot_uniform()
        ],
        KERNEL_INITIALIZER: [
            lecun_uniform(), he_uniform(), glorot_uniform()
        ],
        BIAS_REGULARIZER: [None, l1(), l2(), l1_l2()],
        HIDDEN_LAYER: [
            hidden_layers[NN3_1], hidden_layers[NN3_2], hidden_layers[NN3_3], hidden_layers[NN3_4],
            hidden_layers[DNN5_1], hidden_layers[DNN5_2], hidden_layers[DNN5_3], hidden_layers[DNN5_4],
            hidden_layers[DNN8_1], hidden_layers[DNN8_2], hidden_layers[DNN8_3], hidden_layers[DNN8_4]
        ],
        DROPOUT_RATE: [0, 0.5]
    }
    h = ta.Scan(X_train, y_train, params=p,
                model=fml_model,
                dataset_name=ALL,
                experiment_no=month,
                x_val=X_test, y_val=y_text,
                grid_downsample=0.3,
                clear_tf_session=False)

    # use Scan object as input
    r = ta.Reporting(h)
    # a regression plot for two dimensions
    r.plot_regs()
    # line plot
    r.plot_line()
    # up to two dimensional kernel density estimator
    r.plot_kde('val_acc')
    # a simple histogram
    r.plot_hist(bins=50)
    # heatmap correlation
    r.plot_corr()
    # a four dimensional bar grid
    r.plot_bars('batch_size', 'val_acc', 'first_neuron', 'lr')

    # Clean up the memory
    k.get_session().close()
    k.clear_session()
    tf.reset_default_graph()

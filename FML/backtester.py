# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-21
"""
from model import *
from settings import *
from keras import backend as K

if __name__ == '__main__':
    backtest(param={
        DATA_SET: ALL,
        BATCH_SIZE: 300,
        EPOCHS: 100,
        ACTIVATION: LINEAR,
        BIAS_INITIALIZER: HE_UNIFORM,
        KERNEL_INITIALIZER: GLOROT_UNIFORM,
        BIAS_REGULARIZER: NONE,
        HIDDEN_LAYER: DNN8_2,
        DROPOUT: False,
        DROPOUT_RATE: 0.5
    }, start_number=5, end_number=9, max_pool=5)
    K.tensorflow_backend._get_available_gpus()

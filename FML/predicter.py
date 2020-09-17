# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-21
"""
# noinspection PyUnresolvedReferences
from ensemble import INTERSECTION, GEOMETRIC
from model import *
from settings import *

if __name__ == '__main__':
    get_forward_predict(
        param={
            DATA_SET: ALL,
            BATCH_SIZE: 300,
            EPOCHS: 100,
            ACTIVATION: LINEAR,
            BIAS_INITIALIZER: HE_UNIFORM,
            KERNEL_INITIALIZER: GLOROT_UNIFORM,
            BIAS_REGULARIZER: NONE,
            HIDDEN_LAYER: DNN8_2,
            DROPOUT: False,
            DROPOUT_RATE: 0.5},
        quantile=40, model_num=10, method=GEOMETRIC
    )

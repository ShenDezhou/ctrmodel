#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2019年3月24日

@author: Administrator
'''
import pandas as pd
from tensorflow.python.keras.models import Model
from deepctr.layers.core import PredictionLayer, MLP
from deepctr.input_embedding import create_singlefeat_dict, create_embedding_dict, get_embedding_vec_list, get_inputs_list
from collections import OrderedDict
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import (Concatenate, Dense, Embedding, Flatten,
                                            Input, Reshape, add)
from tensorflow.python.keras.regularizers import l2
from itertools import chain
import keras.utils

def WideDeep():
    # embedding_size=8
    hidden_size=(128, 128)
    l2_reg_linear=1e-5
    l2_reg_embedding=1e-5
    l2_reg_deep=0
    init_std=0.0001
    seed=1024
    keep_prob=1
    activation='relu'
    final_activation='relu'
    
    wide_features=pd.read_csv('data/path_matrix.txt', sep='  ', header=None, nrows=1)
    deep_features=pd.read_csv('data/sns_dense.csv', sep=',', header=0, nrows=2)
    
    wide_input = Input(shape=(wide_features.shape[1],), name='wide_'+str(wide_features.shape[1]))
    wide_term = Dense(1, use_bias=False, activation=None)(wide_input)
    
    deep_input = deep_features.iloc[:,:-1]

    deep_feats = {feat: Input(shape=(1,), name=feat+'_'+str(i)) for i, feat in enumerate(deep_input)}
    deep_list = [v for v in deep_feats.values()]
    deep_input = Concatenate()(deep_list)
    deep_input = Flatten()(deep_input)
    # hidden_size, activation='relu', l2_reg=0, keep_prob=1, use_bn=False, seed=1024
    
    deep_out = MLP(hidden_size=hidden_size, activation=activation, l2_reg=l2_reg_deep, keep_prob=keep_prob, use_bn=False, seed=seed)(deep_input)
    deep_logit = Dense(1, use_bias=False, activation=None)(deep_out)
    final_logit = add([deep_logit, wide_term])
    
    output = PredictionLayer(final_activation)(final_logit)
    
    deep_list.append(wide_input)
    
    model = Model(inputs=deep_list, outputs=output)
    model.summary()
    keras.utils.plot_model(model, to_file='image/widedeep_model.png')
    return model

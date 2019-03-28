#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2019年3月28日

@author: Administrator
'''
from model import WideDeep
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.callbacks import EarlyStopping
from tensorflow.python.framework.dtypes import int32

wide_features=pd.read_csv('data/path_matrix.txt', sep='  ', header=None)
deep_features=pd.read_csv('data/sns_dense.csv', sep=',', header=0)

le = LabelEncoder()
deep_features.iloc[:,0]= le.fit_transform(deep_features.iloc[:,0])

deep_input_feats = deep_features.iloc[:,:-1]
deep_input = [deep_input_feats[v] for v in deep_input_feats]
deep_input.append(wide_features.values)

mms = MinMaxScaler(feature_range=(0, 10000))
final_tags = deep_features.iloc[:,-1:].astype(int)
print(final_tags)
mms.fit(final_tags)
final_tags = mms.transform(final_tags)
print(final_tags)

model = WideDeep()
model.compile(optimizer='adagrad', loss="binary_crossentropy", metrics=[],)

my_callbacks = [EarlyStopping(monitor='loss', min_delta=1e-4, patience=3, verbose=1, mode='min')]
    
model.fit(x=deep_input, y=final_tags, batch_size=64, epochs=100, verbose=1, callbacks=my_callbacks)

pred = model.predict(deep_input, batch_size=2**14)

pred_real = mms.inverse_transform(pred)

for i, x in enumerate(pred_real):
    print(final_tags[i], pred_real[i])
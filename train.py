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
from keras import optimizers,losses
import numpy as np
import pickle
from sklearn.metrics import log_loss, roc_auc_score
import tensorflow as tf
from builtins import int
# from loss import auc

model = WideDeep()
def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
    
#model.compile(optimizer='rmsprop', loss=losses.mse, metrics=["mse"],)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[auc, 'accuracy'])
# 
# wide_features=pd.read_csv('data/path_matrix.txt', sep='  ', header=None)
# deep_features=pd.read_csv('data/sns_dense.csv', sep=',', header=0)
# 
# for index, row in wide_features.iterrows():
#     if not np.any(row):
#         row[:]=8
#         print('normalize')
#         
# maxinterval=deep_features['interval'].max()
# print(maxinterval)
# for index, row in deep_features.iterrows():
#     if row['interval'] == 0:
#         row['interval'] = maxinterval
#         print('interval normalize')

wide_features = pd.read_pickle("prepare/wide.pkl")
deep_features = pd.read_pickle("prepare/deep.pkl")
      
print(deep_features.iloc[0, 0],deep_features.iloc[0, -1])
deep_input_feats = deep_features.iloc[:,:-1]
deep_input = [deep_input_feats[v] for v in deep_input_feats]

# target: box value

target_mms = MinMaxScaler(feature_range=(0, 1))
final_tags = deep_features.iloc[:,[-1]]
final_tags = target_mms.fit_transform(final_tags)

for index, row in enumerate(final_tags):
    if row > 0:
        final_tags[index] = 1
    else:
        final_tags[index] = 0
        
deep_features=deep_features.astype({"boxoffice": int})
print(deep_features.dtypes)

print('nonzero:', np.count_nonzero(final_tags))

# dense features normalization
# http://www.faqs.org/faqs/ai-faq/neural-nets/part2/
for i, feat in enumerate(deep_input_feats):
    print(feat)
    if i==0:
        # id
        le = LabelEncoder()
        deep_input_feats.iloc[:,[0]]= le.fit_transform(deep_features.iloc[:,[0]])
    else:
        input_mms = MinMaxScaler(feature_range=(0, 1))
        deep_input_feats.iloc[:,[i]] = input_mms.fit_transform(deep_input_feats.iloc[:,[i]])
    

deep_input.append(wide_features.values)


my_callbacks = [EarlyStopping(monitor='loss', min_delta=1e-5, patience=3, verbose=1, mode='min')]
model.fit(x=deep_input, y=final_tags, batch_size=64, epochs=100, verbose=1, callbacks=my_callbacks)

pred = model.predict(deep_input, batch_size=2**14)

pred_real = target_mms.inverse_transform(pred)

s0, s1 = 0, 0
for i, x in enumerate(pred_real):
    if pred[i]>0:
        s1+=1
        print(deep_features.iloc[i,0],deep_features.iloc[i,-1], final_tags[i], pred[i])
    else:
        s0+=1
print('s0:',s0,'s1:',s1)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2019年1月22日
@author: Administrator
'''
from PIL import Image
import numpy
import pandas as pd
from builtins import int
import pickle

# matrix_data = numpy.loadtxt('data/path_matrix.txt',dtype=int,delimiter='  ',ndmin=2)
# data = numpy.array(Image.open('data/blank.jpg'))
# 
# print(data.shape)
# row,col,channel=data.shape
# for i in range(row):
#     for j in range(col):
#         #0-7  ~ 0-255
#         data[i,j,0] = (matrix_data[i,j] & 0x01)* 128 +127
#         data[i,j,1] = (matrix_data[i,j] & 0x02)* 128 +127
#         data[i,j,2] = (matrix_data[i,j] & 0x04)* 128 +127
#          
# im = Image.fromarray(data)
# im.save("image/matrix_img.jpg")
# print('process done')
# del matrix_data

wide_features=pd.read_csv('data/path_matrix.txt', sep='  ', header=None)
deep_features=pd.read_csv('data/sns_dense.csv', sep=',', header=0)

for index, row in wide_features.iterrows():
    if not numpy.any(row):
        row[:]=8
        print('normalize')
        
maxinterval=deep_features['interval'].max()
print(maxinterval)
for index, row in deep_features.iterrows():
    if row['interval'] == 0:
        row['interval'] = maxinterval
        print('interval normalize')


wide_features.to_pickle('prepare/wide.pkl', compression='infer', protocol=4)
deep_features.to_pickle('prepare/deep.pkl', compression='infer', protocol=4)

print('done')

# 
# from boxoffice.image.shortestmatrix_alpha import * 
# print(data.shape)
# row,col,channel=data.shape
# for i in range(row):
#     for j in range(col):
#         if matrix_data[i,j] in [0,1]:
#             data[i,j] = [matrix_data[i,j] * 128 +127, 0 ,0,0]
#         if matrix_data[i,j] in [2,3]:
#             data[i,j] = [0 ,(matrix_data[i,j]-2) * 128 +127, 0,0]
#         if matrix_data[i,j] in [4,5]:
#             data[i,j] = [0 ,0,(matrix_data[i,j]-4) * 128 +127,0]
#         if matrix_data[i,j] in [6,7]:
#             data[i,j] = [0,0,0,(matrix_data[i,j]-6) * 0.5 +0.49]
#         
# im = Image.fromarray(data)
# im.save("shortestpath_processed_rgba.png")
# print('process done')
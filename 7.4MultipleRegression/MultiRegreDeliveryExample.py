# -*- coding: utf-8 -*-
"""
@date: 2017-12-04 21:27
@project: MLTest
@file: MultiRegreDeliveryExample.py
@author: yichao0803@gmail.com on PyCharm
"""

from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model

dataPath = r"..\DataSet\Delivery.csv"
deliveryData = genfromtxt(dataPath, delimiter=',')

print("data")
print(deliveryData)

X = deliveryData[:, :-1]
Y = deliveryData[:, -1]

print("X:")
print(X)
print("Y: ")
print(Y)

regr = linear_model.LinearRegression() # 多元回归分析

regr.fit(X, Y)

print("coefficients")
print(regr.coef_) # 全部自变量的斜率 b1,b2,b3
print("intercept: ")
print(regr.intercept_) # 截距 b0

xPred = [[102,6]]
print(xPred)
yPred = regr.predict(xPred)
print("predicted y: ")
print(yPred)
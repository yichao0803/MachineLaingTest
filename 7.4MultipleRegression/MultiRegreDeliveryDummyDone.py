# -*- coding: utf-8 -*-
"""
@date: 2017-12-04 21:26
@project: MLTest
@file: MultiRegreDeliveryDummyDone.py
@author: yichao0803@gmail.com on PyCharm
"""

from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model

dataPath = r"E:\cs\Study\Python\MachineLearningBasics_MachineLaing\Datasets\DeliveryDummyDone.csv"
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

xPred1 = [[102,0,0,1,6]]
xPred2 = [[102,0,1,0,6]]
xPred3 = [[102,1,0,0,6]]
print(xPred1)
print(xPred2)
print(xPred3)
yPred1 = regr.predict(xPred1)
yPred2 = regr.predict(xPred2)
yPred3 = regr.predict(xPred3)
print("predicted y: ")
print(yPred1)
print(yPred2)
print(yPred3)
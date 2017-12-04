# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 15:58:50 2017

@author: Administrator
"""

import numpy as np

def fitSLR(x,y):
    n=len(x)
    dinominator=0
    numerator=0
    for i in range(0,n):
        dinominator +=(x[i]-np.mean(x))**2
        numerator +=(x[i]-np.mean(x))*(y[i]-np.mean(y))
    b1=numerator/dinominator
    b0=np.mean(y)-b1*np.mean(x)
    
    return b0,b1

def predict(x,b0,b1):
    return b0+b1*x

x=[1,3,2,1,3]
y=[14,24,18,17,27]

b0,b1=fitSLR(x,y)
print("intercept: %d slope: %d" %(b0,b1))

x_test=6
y_test=predict(x_test,b0,b1)

print("y_test: %d"% (y_test))
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:31:05 2017

@author: Administrator
"""


from NeuralNetworkSaveWeigths import NeuralNetworkSaveWeigths
import numpy as np

nn = NeuralNetworkSaveWeigths([2,2,1],'xor_weigths', 'tanh')     
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])     
y = np.array([0, 1, 1, 0])     
nn.fit(X, y)     
for i in [[0, 0], [0, 1], [1, 0], [1,1]]:    
    print(i, nn.predict(i))

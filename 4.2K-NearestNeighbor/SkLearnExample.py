# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 18:16:57 2017

@author: Administrator
"""

from sklearn import neighbors
from sklearn import datasets

knn=neighbors.KNeighborsClassifier()

iris=datasets.load_iris()

print(iris)

knn.fit(iris.data,iris.target)
predictedLabel=knn.predict([[0.1,0.2,0.3,0.4]])
print(predictedLabel)
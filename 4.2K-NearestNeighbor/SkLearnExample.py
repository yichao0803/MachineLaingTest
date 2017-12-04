# -*- coding: utf-8 -*-
"""
@date: 2017-12-04 16:20
@project: MLTest
@file: SkLearnExample.py
@author: yichao0803@gmail.com on PyCharm
"""

from sklearn import neighbors
from sklearn import datasets

knn=neighbors.KNeighborsClassifier()

iris=datasets.load_iris()

print(iris)

knn.fit(iris.data,iris.target)
predictedLabel=knn.predict([[0.1,0.2,0.3,0.4]])
print(predictedLabel)
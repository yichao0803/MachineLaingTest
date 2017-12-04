# -*- coding: utf-8 -*-
"""
@date: 2017-12-04 16:20
@project: MLTest
@file: SkLearnSimple.py
@author: yichao0803@gmail.com on PyCharm
"""

from sklearn import svm

X = [[2, 0], [1, 1], [2,3]]
y = [0, 0, 1]
clf = svm.SVC(kernel = 'linear')  # 线性核函数
clf.fit(X, y)  

print(clf)

# get support vectors
print(clf.support_vectors_)

# get indices of support vectors
print(clf.support_)

# get number of support vectors for each class
print(clf.n_support_)

print(clf.predict([[0,0]]))

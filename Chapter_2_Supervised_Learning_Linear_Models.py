#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 14:31:10 2017

@author: sss
"""


import mglearn
import matplotlib.pyplot as plt

# Linear Models
# Trying to learn the parameters w[0] and b on our one-dimensional wave dataset
# might lead to the following line.

#mglearn.plots.plot_linear_regression_wave()



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#X, y = mglearn.datasets.make_wave(n_samples=60)
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
#
#lr = LinearRegression().fit(X_train, y_train)
#
#print("lr.coef_: {}".format(lr.coef_))
#print("lr.intercept_: {}".format(lr.intercept_))
#
# training set and test set performance
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))



## Boston Housing dataset
#X, y = mglearn.datasets.load_extended_boston()
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#lr = LinearRegression().fit(X_train, y_train)
#
#print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
#print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
#
#
## Ridge regression on extended Boston Housing dataset
#from sklearn.linear_model import Ridge
#ridge = Ridge().fit(X_train, y_train)
#print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
#print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
#
#ridge10 = Ridge(alpha=10).fit(X_train, y_train)
#print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
#print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))
#
#ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
#print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
#print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))
#
#plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
#plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
#plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
#
#plt.plot(lr.coef_, 'o', label="LinearRegression")
#plt.xlabel("Coefficient index")
#plt.ylabel("Coefficient magnitude")
#plt.hlines(0, 0, len(lr.coef_))
#plt.ylim(-25, 25)
#plt.legend()



# Subsampled Boston Housing dataset and evaluated LinearRegression and Ridge(
# alpha=1) on subsets of increasing size
mglearn.plots.plot_ridge_n_samples()

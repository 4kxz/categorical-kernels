#!/usr/bin/env python

import json
import os
import sys

# FIXME: Quick path hack.
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from sklearn import cross_validation as cv
from sklearn import grid_search as gs
from sklearn import svm

from kcat.datasets import synthetic
from kcat.kernels.categorical import get_pgen, fast_k0, fast_k1, fast_k2
from kcat.misc import preprocess, GridSearchK0, GridSearchK1

# Generate a dataset and split the data in train and test
X, y = synthetic(320, d=25, c=4)

X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.5, random_state=42)
X_train_pgen = get_pgen(X_train)
X_train_prep =  preprocess(X_train)
X_test_prep = preprocess(X_test)


# Fit

# RBF
clf = svm.SVC(kernel='rbf')
costs = 10.0 ** np.arange(-1, 5)
gammas = 2.0 ** np.arange(-12, -4)
params = dict(C=costs, gamma=gammas)

grid = gs.GridSearchCV(clf, param_grid=params, cv=10)
grid.fit(X_train_prep, y_train)

rbf_fit = grid.best_params_

# K0
costs = 10.0 ** np.arange(-1, 4)
functions = [
    ('ident', 'mean', 'ident'),
    ('ident', 'mean', 'f1'),
    ('ident', 'mean', 'f2'),
    ('ident', 'prod', 'ident'),
    ('ident', 'prod', 'f1'),
    ('ident', 'prod', 'f2'),
    ('f1', 'mean', 'ident'),
    ('f1', 'prod', 'ident'),
]
gammas = 2.0 ** np.arange(-3, 2)

grid = GridSearchK0(functions, gammas, param_grid=dict(C=costs), cv=10)
grid = grid.fit(X_train, y_train)

k0_fit = grid.best_params_

# # K1
# clf = svm.SVC(kernel='precomputed')
# costs = 10.0 ** np.arange(-1, 5)
# functions = [
#     ('ident', 'mean', 'ident'),
#     ('ident', 'mean', 'f1'),
#     ('ident', 'mean', 'f2'),
#     ('ident', 'prod', 'ident'),
#     ('ident', 'prod', 'f1'),
#     ('ident', 'prod', 'f2'),
#     ('f1', 'mean', 'ident'),
#     ('f1', 'prod', 'ident'),
# ]
# gammas = 2.0 ** np.arange(-3, 2)
# alphas = 2.0 ** np.arange(-1, 2)

# grid = GridSearchK1(alphas, functions, gammas, param_grid=dict(C=costs), cv=10)
# grid = grid.fit(X_train, X_train_pgen, y_train)

# k1_fit = grid.best_params_

# # K2
# clf = svm.SVC(kernel='precomputed')
# costs = 10.0 ** np.arange(-1, 4)
# param_grid = dict(C=costs)
# grid = gs.GridSearchCV(clf, param_grid=param_grid, cv=10)
# gram = fast_k2(X_train, X_train, X_train_pgen)
# grid.fit(gram, y_train)

# k2_fit = grid.best_params_

# Evaluation

results = {}

# RBF
clf = svm.SVC(kernel='rbf', **rbf_fit)
clf.fit(X_train_prep, y_train)
y_predict = clf.predict(X_test_prep)

results['rbf'] = (y_predict == y_test).mean()

# K0
ra, rb = k0_fit
clf = svm.SVC(kernel='precomputed', **ra)
clf.fit(fast_k0(X_train, X_train, **rb), y_train)
y_predict = clf.predict(fast_k0(X_test, X_train, **rb))

results['k0'] = (y_predict == y_test).mean()

# # K1
# ra, rb = k1_fit
# clf = svm.SVC(kernel='precomputed', **ra)
# clf.fit(fast_k1(X_train, X_train, X_train_pgen, **rb), y_train)
# y_predict = clf.predict(fast_k1(X_test, X_train, X_train_pgen, **rb))

# results['k1'] = (y_predict == y_test).mean()

# # K2
# clf = svm.SVC(kernel='precomputed', C=10.0)
# clf.fit(fast_k2(X_train, X_train, X_train_pgen), y_train)
# y_predict = clf.predict(fast_k2(X_test, X_train, X_train_pgen))

# results['k2'] = (y_predict == y_test).mean()


# Update file

f = open("./experiments/results/synthetic.json", "r")
data = json.load(f)
f.close()

data.append(results)

f = open("./experiments/results/synthetic.json", "w+")
f.write(json.dumps(data))
f.close()

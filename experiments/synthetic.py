#!/usr/bin/env python

# FIXME: Quick path hack.
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from sklearn import cross_validation as cv
from sklearn import grid_search as gs
from sklearn import svm

from kcat.datasets import synthetic
from kcat.kernels.categorical import fast_k0, fast_k1, fast_k2
from kcat.misc import preprocess

# Fitting the kernels

# Generate a dataset and split the data in train and test
X, y = synthetic(240, d=25, c=4)

X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.5, random_state=42)
X_train_prep =  preprocess(X_train)
X_test_prep = preprocess(X_test)

# Fitting RBF

clf = svm.SVC(kernel='rbf')
costs = 10.0 ** np.arange(-1, 5)
gammas = 2.0 ** np.arange(-12, -4)
params = dict(C=costs, gamma=gammas)

# GridSearch takes care of finding the best parameters using cross-validation:
grid = gs.GridSearchCV(clf, param_grid=params, cv=10)
grid.fit(X_train_prep, y_train)

rbf_results = grid.best_params_
"Best score: {:0.3f}, parameters: {}".format(grid.best_score_, grid.best_params_)

# Fitting K0

clf = svm.SVC(kernel='precomputed')
costs = 10.0 ** np.arange(-1, 4)
# Sensible combinations of prev, composition and post functions.
# The full explicit list for clarity.
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

# Precomputed kernels cannot use GridSearch trivially.
# For now it's done manually.
best_score = 0
best_params = None
for prev, comp, post in functions:
    for g in gammas if prev == 'f1' or post == 'f1' or post == 'f2' else [None]:
        param_dict = dict(prev=prev, comp=comp, post=post, params=dict(gamma=g))
        gram = fast_k0(X_train, **param_dict)
        param_grid = dict(C=costs)
        grid = gs.GridSearchCV(clf, param_grid=param_grid, cv=10)
        grid.fit(gram, y_train)
        if best_score < grid.best_score_:
            best_score = grid.best_score_
            best_params = grid.best_params_, param_dict
        #print("Score: {:0.3f},\tparams: {}\t{}".format(grid.best_score_, param_list, grid.best_params_))

k0_results = best_params
"Best score: {:0.3f}, parameters: {}".format(best_score, best_params)

# Fitting K1

clf = svm.SVC(kernel='precomputed')
costs = 10.0 ** np.arange(-1, 5)
# Sensible combinations of prev, composition and post functions.
# The full explicit list for clarity.
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
alphas = 2.0 ** np.arange(-1, 2)

# Precomputed kernels cannot use GridSearch trivially.
# For now it's done manually.
best_score = 0
best_params = None
for prev, comp, post in functions:
    for g in gammas if prev == 'f1' or post == 'f1' or post == 'f2' else [None]:
        for a in alphas:
            param_dict = dict(prev=prev, comp=comp, post=post, params=dict(gamma=g, alpha=a))
            gram = fast_k1(X_train, **param_dict)
            param_grid = dict(C=costs)
            grid = gs.GridSearchCV(clf, param_grid=param_grid, cv=10)
            grid.fit(gram, y_train)
            if best_score < grid.best_score_:
                best_score = grid.best_score_
                best_params = grid.best_params_, param_dict
            #print("Score: {:0.3f},\tparams: {}\t{}".format(grid.best_score_, param_list, grid.best_params_))

k1_results = best_params
"Best score: {:0.3f}, parameters: {}".format(best_score, best_params)

# Fitting K2

clf = svm.SVC(kernel='precomputed')
costs = 10.0 ** np.arange(-1, 4)
param_grid = dict(C=costs)
grid = gs.GridSearchCV(clf, param_grid=param_grid, cv=10)
gram = fast_k2(X_train)
grid.fit(gram, y_train)

k2_results = grid.best_params_
"Best score: {:0.3f}, parameters: {}".format(grid.best_score_, grid.best_params_)

# Model Evaluation

from kernels.categorical import pmf_from_matrix, k0, k1, k2

pmf = pmf_from_matrix(X_train)

# RBF
clf = svm.SVC(kernel='rbf', **rbf_results)
clf.fit(X_train_prep, y_train)

(clf.predict(X_test_prep) == y_test).mean()

# K0
ra, rb = k0_results
clf = svm.SVC(kernel='precomputed', **ra)
clf.fit(fast_k0(X_train, **rb), y_train)

(clf.predict(k0(X_test, X_train, **rb)) == y_test).mean()

# K1
ra, rb = k1_results
clf = svm.SVC(kernel='precomputed', **ra)
clf.fit(fast_k1(X_train, pmf=pmf, **rb), y_train)

(clf.predict(k1(X_test, X_train, pmf=pmf, **rb)) == y_test).mean()

clf = svm.SVC(kernel='precomputed', C=10.0)
clf.fit(fast_k2(X_train, pmf=pmf), y_train)

(clf.predict(k2(X_test, X_train, pmf=pmf)) == y_test).mean()

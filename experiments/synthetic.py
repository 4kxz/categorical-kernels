#!/usr/bin/env python

import json
import os
import sys
import time

# FIXME: Quick path hack.
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from sklearn import cross_validation as cv
from sklearn import grid_search as gs
from sklearn import svm

from kcat.datasets import synthetic
from kcat.kernels.categorical import get_pgen, fast_k0, fast_k1, fast_k2
from kcat.misc import GridSearchK0, GridSearchK1


# Load data:
f = open("./experiments/results/synthetic.json", "r")
data = json.load(f)
f.close()


# Find next available ID:
lastid = -1
for d in data:
    lastid = d['meta']['id'] if lastid < d['meta']['id'] else lastid
nextid = lastid + 1


# Generate a dataset:
data_args = dict(m=150, n=25, c=4, p=0.5, random_state=nextid)
X, y, bincoder = synthetic(**data_args)
# Split the data in train and test:
split = cv.train_test_split(X, y, test_size=0.5, random_state=nextid)
X_train, X_test, y_train, y_test = split
Xb_train =  bincoder(X_train)
Xb_test = bincoder(X_test)
pgen = get_pgen(X_train)
# Specify the cross-validation to use
cvf = cv.StratifiedKFold(y_train, 5)


# Fit

# RBF
clf = svm.SVC(kernel='rbf')
costs = 10.0 ** np.arange(-1, 8)
gammas = 2.0 ** np.arange(-15, 1)
params = dict(C=costs, gamma=gammas)

grid = gs.GridSearchCV(clf, param_grid=params, cv=cvf)
grid.fit(Xb_train, y_train)

rbf_fit = grid.best_params_

# K0
costs = 10.0 ** np.arange(-1, 8)
functions = [
    ('ident', 'ident'),
    ('ident', 'f1'),
    ('f1', 'ident'),
]
gammas = 2.0 ** np.arange(-6, 5)
params = dict(C=costs)

grid = GridSearchK0(functions, gammas, param_grid=params, cv=cvf)
grid = grid.fit(X_train, y_train)

k0_fit = grid.best_params_

# K1
clf = svm.SVC(kernel='precomputed')
costs = 10.0 ** np.arange(-1, 8)
functions = [
    ('ident', 'ident'),
    ('ident', 'f1'),
    ('ident', 'f2'),
    ('f1', 'ident'),
]
gammas = 2.0 ** np.arange(-6, 4)
alphas = 2.0 ** np.arange(-4, 4)
params = dict(C=costs)

grid = GridSearchK1(alphas, functions, gammas, param_grid=params, cv=cvf)
grid = grid.fit(X_train, pgen, y_train)

k1_fit = grid.best_params_

# K2
clf = svm.SVC(kernel='precomputed')
costs = 10.0 ** np.arange(-1, 8)
params = dict(C=costs)

grid = gs.GridSearchCV(clf, param_grid=params, cv=cvf)
gram = fast_k2(X_train, X_train, pgen)
grid.fit(gram, y_train)

k2_fit = grid.best_params_

# Evaluation

results = {}

# RBF
clf = svm.SVC(kernel='rbf', **rbf_fit)
clf.fit(Xb_train, y_train)
y_predict = clf.predict(Xb_test)

results['rbf'] = {'params': rbf_fit, 'score': (y_predict == y_test).mean()}

# K0
ra, rb = k0_fit
clf = svm.SVC(kernel='precomputed', **ra)
clf.fit(fast_k0(X_train, X_train, **rb), y_train)
y_predict = clf.predict(fast_k0(X_test, X_train, **rb))

results['k0'] = {'params': k0_fit, 'score': (y_predict == y_test).mean()}

# K1
ra, rb = k1_fit
clf = svm.SVC(kernel='precomputed', **ra)
clf.fit(fast_k1(X_train, X_train, pgen, **rb), y_train)
y_predict = clf.predict(fast_k1(X_test, X_train, pgen, **rb))

results['k1'] = {'params': k1_fit, 'score': (y_predict == y_test).mean()}

# K2
clf = svm.SVC(kernel='precomputed', C=10.0)
clf.fit(fast_k2(X_train, X_train, pgen), y_train)
y_predict = clf.predict(fast_k2(X_test, X_train, pgen))

results['k2'] = {'params': k2_fit, 'score': (y_predict == y_test).mean()}


# Update file

results['id'] = nextid
results['data_args'] = data_args
results['timestamp'] = time.asctime()

data.append(results)

f = open("./experiments/results/synthetic.json", "w+")
f.write(json.dumps(data))
f.close()

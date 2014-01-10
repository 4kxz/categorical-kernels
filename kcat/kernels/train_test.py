"""This module has a helper class to avoid boilerplate code to train
and test the various models.

Instead of having to write a script each time, the same can be achieved
by changing the attributes of the helper class.

For example, in order to change the default search space for grid search
with the kernel k1 it would suffice to change the *default_params*
attribute in TrainTestK1.
"""

import collections

import numpy as np
from sklearn import svm

from . import functions as fn
from . import grid_search as gs

np.set_printoptions(precision=2, threshold=4, edgeitems=2)

# Training boilerplate

class CustomTrainTest:
    svc = None
    kernel = None
    search_class = None
    default_params = None

    @classmethod
    def train(cls, cv, **kwargs):
        clf = svm.SVC(cls.svc, max_iter=2**20)
        search = cls.search_class(estimator=clf, cv=cv, **cls.default_params)
        search.fit(**kwargs)
        return search

    @classmethod
    def test(cls, search, X, y, **kwargs):
        if cls.kernel is None:
            m = X
        else:
            kwargs.update(search.best_kparams_)
            m = cls.kernel(X, search.X, **kwargs)
        prediction = search.best_estimator_.predict(m)
        results = {}
        results.update(search.details)
        results.update({'test_score': (prediction == y).mean()})
        return results

    @classmethod
    def train_test(cls, cv, X_train, y_train, X_test, y_test, **kwargs):
        search = cls.train(cv=cv, X=X_train, y=y_train, **kwargs)
        return cls.test(search=search, X=X_test, y=y_test, **kwargs)


class TrainTestRBF(CustomTrainTest):
    svc = 'rbf'
    kernel = None
    search_class = gs.CustomGridSearch
    default_params = {
        'C': 10.0 ** np.arange(-1, 3),
        'gamma': 2.0 ** np.arange(-12, 1),
        }


class TrainTestK0(CustomTrainTest):
    svc = 'precomputed'
    kernel = fn.fast_k0
    search_class = gs.GridSearchK0
    default_params = {
        'C': 10.0 ** np.arange(-1, 3),
        'functions': [
            ('ident', 'ident'),
            ('ident', 'f1'),
            ('f1', 'ident'),
            ],
        'gamma': 2.0 ** np.arange(-3, 3),
        }


class TrainTestK1(CustomTrainTest):
    svc = 'precomputed'
    kernel = fn.fast_k1
    search_class = gs.GridSearchK1
    default_params = {
        'C': 10.0 ** np.arange(-1, 3),
        'alpha': 1.5 ** np.arange(-4, 3),
        'functions': [
            ('ident', 'ident'),
            ('ident', 'f1'),
            ('ident', 'f2'),
            ('f1', 'ident'),
            ],
        'gamma': 2.0 ** np.arange(-3, 3),
        }


class TrainTestK2(CustomTrainTest):
    svc = 'precomputed'
    kernel = fn.fast_k2
    search_class = gs.GridSearchK2
    default_params = {
        'C': 10.0 ** np.arange(-1, 3),
        'functions': [
            ('ident', 'ident'),
            ('ident', 'f1'),
            ('ident', 'f2'),
            ('f1', 'ident'),
            ],
        'gamma': 2.0 ** np.arange(-3, 1),
        }


class TrainTestM1(CustomTrainTest):
    svc = 'precomputed'
    kernel = fn.fast_m1
    search_class = gs.GridSearchM1
    default_params = {
        'C': 10.0 ** np.arange(-1, 3),
        'alpha': 1.5 ** np.arange(-4, 3),
        }


class TrainTestELK(CustomTrainTest):
    svc = 'precomputed'
    kernel = fn.elk
    search_class = gs.GridSearchELK
    default_params = {
        'C': 10.0 ** np.arange(-1, 3),
        }

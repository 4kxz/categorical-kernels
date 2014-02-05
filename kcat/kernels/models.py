"""This module has a helper class to avoid boilerplate code to train
and test the various models.

Instead of having to write a script each time, the same can be achieved
by changing the attributes of the helper class.

For example, in order to change the default search space for grid search
with the kernel k1 it would suffice to change the *default_params*
attribute in K1.
"""

import collections

import numpy as np
from sklearn import svm

from . import functions as fn
from . import grid_search as gs
from ..utils import get_pgen

np.set_printoptions(precision=2, threshold=4, edgeitems=2)

# Training boilerplate

class Model:
    svc = None
    kernel = None
    data = None
    searcher = None
    default_params = None

    @classmethod
    def train(cls, cv, X, y, **kwargs):
        X = X['categorical'] if cls.data == 'categorical' else X['default']
        clf = svm.SVC(cls.svc, max_iter=2**20)
        search = cls.searcher(estimator=clf, cv=cv, **cls.default_params)
        search.fit(X=X, y=y, **kwargs)
        return search

    @classmethod
    def test(cls, search, X, y, **kwargs):
        X = X['categorical'] if cls.data == 'categorical' else X['default']
        prediction = search.predict(X=X)
        results = {'test_score': np.mean(prediction == y)}
        results.update(search.details)
        return results


class RBF(Model):
    svc = 'rbf'
    kernel = None
    searcher = gs.GridSearchWrapper
    default_params = {
        'C': 10.0 ** np.arange(-1, 3),
        'gamma': 2.0 ** np.arange(-12, 1),
        }


class K0(Model):
    data = 'categorical'
    svc = 'precomputed'
    kernel = fn.fast_k0
    searcher = gs.GridSearchK0
    default_params = {
        'C': 10.0 ** np.arange(-1, 3),
        'functions': [
            ('ident', 'ident'),
            ('ident', 'f1'),
            ('f1', 'ident'),
            ],
        'gamma': 2.0 ** np.arange(-3, 3),
        }


class K1(Model):
    data = 'categorical'
    svc = 'precomputed'
    kernel = fn.fast_k1
    searcher = gs.GridSearchK1
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


class K2(Model):
    data = 'categorical'
    svc = 'precomputed'
    kernel = fn.fast_k2
    searcher = gs.GridSearchK2
    default_params = {
        'C': 10.0 ** np.arange(-1, 3),
        'functions': [
            ('ident', 'ident'),
            ('ident', 'f1'),
            ('ident', 'f2'),
            ('f1', 'ident'),
            ],
        'gamma': 2.0 ** np.arange(-3, 2),
        }


class M1(Model):
    data = 'categorical'
    svc = 'precomputed'
    kernel = fn.m1
    searcher = gs.GridSearchM1
    default_params = {
        'C': 10.0 ** np.arange(-1, 3),
        'alpha': 1.5 ** np.arange(-4, 3),
        'functions': [
            ('ident', 'ident'),
            ('ident', 'f1'),
            ('f1', 'ident'),
            ],
        'gamma': 2.0 ** np.arange(-3, 3),
        }


class ELK(Model):
    data = 'quantitative'
    svc = 'precomputed'
    kernel = fn.elk
    searcher = gs.GridSearchELK
    default_params = {
        'C': 10.0 ** np.arange(-1, 3),
        }


train_test_models = (RBF, K0, K1, K2, M1, ELK)

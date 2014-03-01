"""This module contains helper classes to avoid boilerplate code to
train and test the various models.
"""

import collections

import numpy as np
from sklearn import svm

from . import search as ks
from ..utils import get_pgen

# Nicer output in Sphinx
np.set_printoptions(precision=2, threshold=4, edgeitems=2)


class BaseHelper:
    svc = None
    data = None
    searcher = None
    default_params = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We don't want instances overwritting the class default params.
        self.default_params = self.default_params.copy()
        self.name = self.__class__.__name__

    def train(self, cv, X, y, **kwargs):
        X = X['categorical'] if self.data == 'categorical' else X['default']
        clf = svm.SVC(self.svc, max_iter=2**20)
        search = self.searcher(estimator=clf, cv=cv, **self.default_params)
        search.fit(X=X, y=y, **kwargs)
        return search

    def test(self, search, X, y, **kwargs):
        X = X['categorical'] if self.data == 'categorical' else X['default']
        prediction = search.predict(X=X)
        results = {'test_score': np.mean(prediction == y)}
        results.update(search.details)
        return results


class ELK(BaseHelper):
    data = 'quantitative'
    svc = 'precomputed'
    searcher = ks.ELKSearch
    default_params = {
        'C': 10.0 ** np.arange(-1, 3),
        }


class K0(BaseHelper):
    data = 'categorical'
    svc = 'precomputed'
    searcher = ks.K0Search
    default_params = {
        'C': 10.0 ** np.arange(-1, 3),
        'functions': [
            ('ident', 'ident'),
            ('ident', 'f1'),
            ('f1', 'ident'),
            ],
        'gamma': 2.0 ** np.arange(-3, 3),
        }


class K1(BaseHelper):
    data = 'categorical'
    svc = 'precomputed'
    searcher = ks.K1Search
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


class K2(BaseHelper):
    data = 'categorical'
    svc = 'precomputed'
    searcher = ks.K2Search
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


class M1(BaseHelper):
    data = 'categorical'
    svc = 'precomputed'
    searcher = ks.M1Search
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


class M2(M1):
    searcher = ks.M2Search


class M3(M1):
    searcher = ks.M3Search


class M4(M1):
    searcher = ks.M4Search


class RBF(BaseHelper):
    data = 'quantitative'
    svc = 'rbf'
    searcher = ks.RBFSearch
    default_params = {
        'C': 10.0 ** np.arange(-1, 3),
        'gamma': 2.0 ** np.arange(-12, 1),
        }


DEFAULT_MODELS = (RBF, K0, K1, K2, ELK, M1, M2, M3, M4)

"""Boilerplate to train and test various models."""

import collections

from sklearn import svm

from . import functions as fn
from . import grid_search as gs
from . import parameters as PARAM


# Training boilerplate

class CustomTrainTest:
    svc = None
    kernel = None
    search_class = None
    search_params = None

    @classmethod
    def train(cls, cv, **kwargs):
        clf = svm.SVC(cls.svc, max_iter=2**20)
        search = cls.search_class(clf=clf, cv=cv, **cls.search_params)
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
        return (prediction == y).mean()

    @classmethod
    def train_test(cls, cv, X_train, y_train, X_test, y_test):
        search = cls.train(cv=cv, X=X_train, y=y_train)
        return cls.test(search=search, X=X_test, y=y_test)


class TrainTestRBF(CustomTrainTest):
    svc = 'rbf'
    kernel = None
    search_class = gs.CustomGridSearch
    search_params = PARAM.RBF


class TrainTestK0(CustomTrainTest):
    svc = 'precomputed'
    kernel = fn.fast_k0
    search_class = gs.GridSearchK0
    search_params = PARAM.K0

    @classmethod
    def train_test(cls, cv, X_train, y_train, X_test, y_test):
        search = cls.train(cv=cv, X=X_train, y=y_train)
        return cls.test(search=search, X=X_test, y=y_test)


class TrainTestK1(CustomTrainTest):
    svc = 'precomputed'
    kernel = fn.fast_k1
    search_class = gs.GridSearchK1
    search_params = PARAM.K1

    @classmethod
    def train_test(cls, cv, X_train, y_train, X_test, y_test, pgen):
        model = cls.train(cv=cv, X=X_train, y=y_train, pgen=pgen)
        return cls.test(model, X=X_test, y=y_test, pgen=pgen)


class TrainTestK2(CustomTrainTest):
    svc = 'precomputed'
    kernel = fn.fast_k2
    search_class = gs.GridSearchK2
    search_params = PARAM.K2

    @classmethod
    def train_test(cls, cv, X_train, y_train, X_test, y_test, pgen):
        model = cls.train(cv=cv, X=X_train, y=y_train, pgen=pgen)
        return cls.test(model, X=X_test, y=y_test, pgen=pgen)


class TrainTestM1(CustomTrainTest):
    svc = 'precomputed'
    kernel = fn.fast_m1
    search_class = gs.GridSearchM1
    search_params = PARAM.M1


class TrainTestELK(CustomTrainTest):
    svc = 'precomputed'
    kernel = fn.elk
    search_class = gs.GridSearchELK
    search_params = PARAM.ELK

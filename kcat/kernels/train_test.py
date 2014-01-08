"""Boilerplate to train and test various models."""

import collections

from sklearn import svm

from . import functions as fn
from . import grid_search as gs
from . import parameters as pm


# Training boilerplate

Fit = collections.namedtuple('Fit', 'estimator, params, score')

max_it = 2**20

def train_rbf(Xb_train, y_train, cvf):
    clf = svm.SVC(kernel='rbf', max_iter=max_it)
    result = gs.GridSearchCV(clf, cv=cvf, **pm.PARAMS_RBF)
    result.fit(Xb_train, y_train)
    return Fit(result.best_estimator_, result.best_params_, result.best_score_)

def train_k0(X_train, y_train, cvf):
    clf = svm.SVC(kernel='precomputed', max_iter=max_it)
    result = gs.GridSearchK0(clf, cv=cvf, **pm.PARAMS_K0)
    result.fit(X_train, y_train)
    return Fit(result.best_estimator_, result.best_params_, result.best_score_)

def train_k1(X_train, y_train, pgen, cvf):
    clf = svm.SVC(kernel='precomputed', max_iter=max_it)
    result = gs.GridSearchK1(clf, cv=cvf, **pm.PARAMS_K1)
    result.fit(X_train, y_train, pgen)
    return Fit(result.best_estimator_, result.best_params_, result.best_score_)

def train_k2(X_train, y_train, pgen, cvf):
    clf = svm.SVC(kernel='precomputed', max_iter=max_it)
    result = gs.GridSearchK2(clf, cv=cvf, **pm.PARAMS_K2)
    result.fit(X_train, y_train, pgen)
    return Fit(result.best_estimator_, result.best_params_, result.best_score_)

def train_m0(X_train, y_train, pgen, cvf):
    clf = svm.SVC(kernel='precomputed', max_iter=max_it)
    result = gs.GridSearchM0(clf, cv=cvf, **pm.PARAMS_M0)
    result.fit(X_train, y_train, pgen)
    return Fit(result.best_estimator_, result.best_params_, result.best_score_)

def train_m1(X_train, y_train, pgen, cvf):
    clf = svm.SVC(kernel='precomputed', max_iter=max_it)
    result = gs.GridSearchM1(clf, cv=cvf, **pm.PARAMS_M1)
    result.fit(X_train, y_train, pgen)
    return Fit(result.best_estimator_, result.best_params_, result.best_score_)

def train_elk(X_train, y_train, Xpgen, cvf):
    clf = svm.SVC(kernel='precomputed', max_iter=max_it)
    result = gs.GridSearchELK(clf, cv=cvf, **pm.PARAMS_ELK)
    result.fit(X_train, y_train, Xpgen)
    return Fit(result.best_estimator_, result.best_params_, result.best_score_)


# Testing boilerplate

def test_rbf(clf, Xb_test, y_test):
    prediction = clf.predict(Xb_test)
    return (prediction == y_test).mean()

def test_k0(clf, X_train, X_test, y_test, params):
    gram = fn.fast_k0(X_test, X_train, **params)
    prediction = clf.predict(gram)
    return (prediction == y_test).mean()

def test_k1(clf, X_train, X_test, y_test, pgen, params):
    gram = fn.fast_k1(X_test, X_train, pgen, **params)
    prediction = clf.predict(gram)
    return (prediction == y_test).mean()

def test_k2(clf, X_train, X_test, y_test, pgen, params):
    gram = fn.fast_k2(X_test, X_train, pgen, **params)
    prediction = clf.predict(gram)
    return (prediction == y_test).mean()

def test_m0(clf, X_train, X_test, y_test, pgen, params):
    gram = fn.fast_m0(X_test, X_train, pgen, **params)
    prediction = clf.predict(gram)
    return (prediction == y_test).mean()

def test_m1(clf, X_train, X_test, y_test, pgen, params):
    gram = fn.fast_m1(X_test, X_train, pgen, **params)
    prediction = clf.predict(gram)
    return (prediction == y_test).mean()

def test_elk(clf, X_train, X_test, y_test, X_train_pgen, X_test_pgen):
    gram = fn.elk(X_train, X_test, X_train_pgen, X_test_pgen)
    prediction = clf.predict(gram)
    return (prediction == y_test).mean()

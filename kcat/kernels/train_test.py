import collections

from sklearn import svm

from .functions import fast_k0, fast_k1, fast_k2
from .grid_search import GridSearchCV, GridSearchK0, GridSearchK1, GridSearchK2
from .parameters import PARAMS_RBF, PARAMS_K0, PARAMS_K1, PARAMS_K2

#------------------------------------------------------------------------------
# Training boilerplate
#------------------------------------------------------------------------------

Fit = collections.namedtuple('Fit', 'estimator, params, score')

def train_rbf(Xb_train, y_train, cvf):
    clf = svm.SVC(kernel='rbf')
    result = GridSearchCV(clf, cv=cvf, **PARAMS_RBF)
    result.fit(Xb_train, y_train)
    return Fit(result.best_estimator_, result.best_params_, result.best_score_)

def train_k0(X_train, y_train, cvf):
    clf = svm.SVC(kernel='precomputed')
    result = GridSearchK0(clf, cv=cvf, **PARAMS_K0)
    result.fit(X_train, y_train)
    return Fit(result.best_estimator_, result.best_params_, result.best_score_)

def train_k1(X_train, y_train, pgen, cvf):
    clf = svm.SVC(kernel='precomputed')
    result = GridSearchK1(clf, cv=cvf, **PARAMS_K1)
    result.fit(X_train, y_train, pgen)
    return Fit(result.best_estimator_, result.best_params_, result.best_score_)

def train_k2(X_train, y_train, pgen, cvf):
    clf = svm.SVC(kernel='precomputed')
    result = GridSearchK2(clf, cv=cvf, **PARAMS_K2)
    result.fit(X_train, y_train, pgen)
    return Fit(result.best_estimator_, result.best_params_, result.best_score_)


#------------------------------------------------------------------------------
# Testing boilerplate
#------------------------------------------------------------------------------

def test_rbf(clf, Xb_test, y_test):
    prediction = clf.predict(Xb_test)
    return (prediction == y_test).mean()

def test_k0(clf, X_train, X_test, y_test, params):
    gram = fast_k0(X_test, X_train, **params)
    prediction = clf.predict(gram)
    return (prediction == y_test).mean()

def test_k1(clf, X_train, X_test, y_test, pgen, params):
    gram = fast_k1(X_test, X_train, pgen, **params)
    prediction = clf.predict(gram)
    return (prediction == y_test).mean()

def test_k2(clf, X_train, X_test, y_test, pgen):
    gram = fast_k2(X_test, X_train, pgen)
    prediction = clf.predict(gram)
    return (prediction == y_test).mean()

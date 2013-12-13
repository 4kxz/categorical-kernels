"""
Classes to use Grid Search on the custom kernels defined in
:mod:`kcat.kernels.functions`.

Thir interface is very similar to `GridSearchCV
<http://scikit-learn.org/stable/modules/generated/\
sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV>`_
of scikit-learn, and most parameters from GridSearchCV can be used.
"""

from sklearn.grid_search import GridSearchCV

from .functions import fast_k0, fast_k1, fast_k2


class GridSearchK0:
    """
    Find best parameters for *K0*.

    :param functions: A list with the 'prev' and 'post' functions.
    :param gammas: A list of values.
    """

    def __init__(self, clf, functions, gammas, **kwargs):
        self.clf = clf
        self.functions = functions
        self.gammas = gammas
        self.params = kwargs
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = 0

    def fit(self, X, y):
        """
        Fit the model to the data matrix *X* and class vector *y*.
        """
        for prev, post in self.functions:
            uses_gammas = prev == 'f1' or post in ('f1', 'f2')
            for g in self.gammas if uses_gammas else [None]:
                result = GridSearchCV(self.clf, **self.params)
                params = dict(prev=prev, post=post, gamma=g)
                gram = fast_k0(X, X, **params)
                result.fit(gram, y)
                if result.best_score_ >= self.best_score_:
                    self.best_estimator_ = result.best_estimator_
                    self.best_params_ = (result.best_params_, params)
                    self.best_score_ = result.best_score_


class GridSearchK1:
    """
    Find best parameters for *K1*.

    :param alphas: A list of values.
    :param functions: A list with the 'prev' and 'post' functions.
    :param gammas: A list of values.
    """

    def __init__(self, clf, alphas, functions, gammas, **kwargs):
        self.clf = clf
        self.alphas = alphas
        self.functions = functions
        self.gammas = gammas
        self.params = kwargs
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = 0

    def fit(self, X, y, pgen):
        """
        Fit the model to the data matrix *X* and class vector *y*. *pgen* is
        a probability distribution, see :meth:`~kcat.kernels.utils.get_pgen`.
        """
        for prev, post in self.functions:
            uses_gammas = prev == 'f1' or post in ('f1', 'f2')
            for g in self.gammas if uses_gammas else [None]:
                for a in self.alphas:
                    result = GridSearchCV(self.clf, **self.params)
                    params = dict(alpha=a, prev=prev, post=post, gamma=g)
                    gram = fast_k1(X, X, pgen, **params)
                    result.fit(gram, y)
                    if result.best_score_ >= self.best_score_:
                        self.best_estimator_ = result.best_estimator_
                        self.best_params_ = (result.best_params_, params)
                        self.best_score_ = result.best_score_


class GridSearchK2:
    """Find best parameters for *K2*."""

    def __init__(self, clf, functions, gammas, **kwargs):
        self.clf = clf
        self.functions = functions
        self.gammas = gammas
        self.params = kwargs
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = 0

    def fit(self, X, y, pgen):
        """
        Fit the model to the data matrix *X* and class vector *y*. *pgen* is
        a probability distribution, see :meth:`~kcat.kernels.utils.get_pgen`.
        """
        for prev, post in self.functions:
            uses_gammas = prev == 'f1' or post in ('f1', 'f2')
            for g in self.gammas if uses_gammas else [None]:
                result = GridSearchCV(self.clf, **self.params)
                params = dict(prev=prev, post=post, gamma=g)
                gram = fast_k2(X, X, pgen, **params)
                result.fit(gram, y)
                if result.best_score_ >= self.best_score_:
                    self.best_estimator_ = result.best_estimator_
                    self.best_params_ = (result.best_params_, params)
                    self.best_score_ = result.best_score_

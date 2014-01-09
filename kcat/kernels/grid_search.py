"""Classes to use Grid Search on the custom kernels defined in
:mod:`kcat.kernels.functions`.

Thir interface is very similar to `GridSearchCV
<http://scikit-learn.org/stable/modules/generated/\
sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV>`_
of scikit-learn, and most parameters from GridSearchCV can be used.
"""

from sklearn.grid_search import GridSearchCV

from . import functions as fn


class CustomGridSearch:
    """Base class for custom grid searches."""

    def __init__(self, clf, cv, **kwargs):
        self.gskwargs = {
            'estimator': clf,
            'cv': cv,
            'param_grid': kwargs,
            }
        self.best_score_ = 0
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_kparams_ = None
        self.X = None

    def fit(self, X, y):
        """Fit the model to the data matrix *X* and class vector *y*."""
        result = GridSearchCV(**self.gskwargs)
        result.fit(X, y)
        self.best_estimator_ = result.best_estimator_
        self.best_params_ = result.best_params_
        self.best_score_ = result.best_score_


class GridSearchK0(CustomGridSearch):
    """Finds the best parameters for *K0*.

    :param functions: A list with the 'prev' and 'post' functions.
    :param gamma: A list of values.
    """

    def __init__(self, functions, gamma, **kwargs):
        self.functions = functions
        self.gamma = gamma
        super().__init__(**kwargs)

    def fit(self, X, y):
        """Fit the model to the data matrix *X* and class vector *y*."""
        self.X = X
        for prev, post in self.functions:
            uses_gammas = prev == 'f1' or post in ('f1', 'f2')
            for g in self.gamma if uses_gammas else [None]:
                result = GridSearchCV(**self.gskwargs)
                params = dict(prev=prev, post=post, gamma=g)
                gram = fn.fast_k0(X, X, **params)
                result.fit(gram, y)
                if result.best_score_ >= self.best_score_:
                    self.best_score_ = result.best_score_
                    self.best_estimator_ = result.best_estimator_
                    self.best_params_ = result.best_params_
                    self.best_kparams_ = params


class GridSearchK1(CustomGridSearch):
    """Finds the best parameters for *K1*.

    :param alpha: A list of values.
    :param functions: A list with the 'prev' and 'post' functions.
    :param gamma: A list of values.
    """

    def __init__(self, alpha, functions, gamma, **kwargs):
        self.alpha = alpha
        self.functions = functions
        self.gamma = gamma
        super().__init__(**kwargs)

    def fit(self, X, y, pgen):
        """Fit the model to the data matrix *X* and class vector *y*.
        *pgen* is a probability distribution, see
        :meth:`~kcat.kernels.utils.get_pgen`.
        """
        self.X = X
        for prev, post in self.functions:
            uses_gammas = prev == 'f1' or post in ('f1', 'f2')
            for g in self.gamma if uses_gammas else [None]:
                for a in self.alpha:
                    result = GridSearchCV(**self.gskwargs)
                    params = dict(alpha=a, prev=prev, post=post, gamma=g)
                    gram = fn.fast_k1(X, X, pgen, **params)
                    result.fit(gram, y)
                    if result.best_score_ >= self.best_score_:
                        self.best_score_ = result.best_score_
                        self.best_estimator_ = result.best_estimator_
                        self.best_params_ = result.best_params_
                        self.best_kparams_ = params


class GridSearchK2(CustomGridSearch):
    """Finds the best parameters for *K2*."""

    def __init__(self, functions, gamma, **kwargs):
        self.functions = functions
        self.gamma = gamma
        super().__init__(**kwargs)

    def fit(self, X, y, pgen):
        """Fit the model to the data matrix *X* and class vector *y*.
        *pgen* is a probability distribution, see
        :meth:`~kcat.kernels.utils.get_pgen`.
        """
        self.X = X
        for prev, post in self.functions:
            uses_gammas = prev == 'f1' or post in ('f1', 'f2')
            for g in self.gamma if uses_gammas else [None]:
                result = GridSearchCV(**self.gskwargs)
                params = dict(prev=prev, post=post, gamma=g)
                gram = fn.fast_k2(X, X, pgen, **params)
                result.fit(gram, y)
                if result.best_score_ >= self.best_score_:
                    self.best_score_ = result.best_score_
                    self.best_estimator_ = result.best_estimator_
                    self.best_params_ = result.best_params_
                    self.best_kparams_ = params


class GridSearchM1(CustomGridSearch):
    """Finds the best parameters for *M1*."""

    def __init__(self, clf, alpha, **kwargs):
        self.alpha = alpha
        super().__init__(**kwargs)

    def fit(self, X, y, pgen):
        """Fit the model to the data matrix *X* and class vector *y*.
        *pgen* is a probability distribution, see
        :meth:`~kcat.kernels.utils.get_pgen`.
        """
        self.X = X
        for a in self.alpha:
            result = GridSearchCV(**self.gskwargs)
            params = dict(alpha=a)
            gram = fn.fast_k1(X, X, pgen, **params)
            result.fit(gram, y)
            if result.best_score_ >= self.best_score_:
                self.best_score_ = result.best_score_
                self.best_estimator_ = result.best_estimator_
                self.best_params_ = result.best_params_
                self.best_kparams_ = params


class GridSearchELK(CustomGridSearch):
    """Finds the best parameters for *ELK*."""

    def fit(self, X, y, Xpgen):
        """Fit the model to the data matrix *X* and class vector *y*."""
        self.X = X
        result = GridSearchCV(**self.gskwargs)
        gram = fn.elk(X, X, Xpgen, Xpgen)
        result.fit(gram, y)
        self.best_estimator_ = result.best_estimator_
        self.best_params_ = result.best_params_
        self.best_score_ = result.best_score_

"""Classes to perform Grid Search on the custom kernels defined in
:mod:`kcat.kernels.functions`.

Their interface is very similar to scikit-learn's
`GridSearchCV <http://scikit-learn.org/stable/modules/generated/sklearn\
.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV>`_,
and the same parameters should be used.
"""

from sklearn.grid_search import GridSearchCV

from . import functions as kf
from ..utils import pgen


class BaseSearch:
    """Base class. Can be extended to do custom searches.

    Any subclass should deal with kernel specific keyword arguments such
    as *alpha*, *gamma*, *prev*, etc.
    Common arguments like *estimator*, *cv*, *C* and so on are handled
    by calling `super()`.
    """

    def __init__(self, estimator, cv, **kwargs):
        self.gskwargs = {
            'estimator': estimator,
            'cv': cv,
            'param_grid': kwargs,
            }
        self.best_score_ = 0
        self.best_params_ = {}
        self.best_kparams_ = {}
        self.best_estimator_ = None
        self.X = None
        self.pgen = None

    def fit(self, X, y):
        """Fit the model to the data matrix *X* and class vector *y*.

        Args:
            X: Numpy matrix with the examples in rows.
            y: Numpy array with the class of each example.
        """
        search = GridSearchCV(**self.gskwargs)
        search.fit(X, y)
        self.best_estimator_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_

    def predict(self, X):
        """
        Args:
            X: Numpy matrix with the examples in rows.

        Returns:
            A Numpy vector with the predicted classes.
        """
        return self.best_estimator_.predict(X)

    @property
    def details(self):
        """A dictionary with the found parameters and error."""
        details = {
            'train_score': self.best_score_,
            'best_parameters': {},
            }
        details['best_parameters'].update(self.best_params_)
        details['best_parameters'].update(self.best_kparams_)
        return details


class SearchK0(BaseSearch):
    """Finds the best parameters for :meth:`kcat.kernels.functoins.k0`.

    Args:
        functions: A list with tuples of the form ('prev', 'post').
        gamma: A list of floats with the gamma values.
    """

    def __init__(self, functions, gamma, **kwargs):
        self.functions = functions
        self.gamma = gamma
        super().__init__(**kwargs)

    def fit(self, X, y):
        self.X = X
        # Only 'f1' and 'f2' use gammas, no need to search all the
        # permutations.
        for prev, post in self.functions:
            uses_gammas = prev == 'f1' or post in ('f1', 'f2')
            for g in self.gamma if uses_gammas else [None]:
                search = GridSearchCV(**self.gskwargs)
                params = dict(prev=prev, post=post, gamma=g)
                gram = kf.k0(X, X, **params)
                search.fit(gram, y)
                if search.best_score_ >= self.best_score_:
                    self.best_score_ = search.best_score_
                    self.best_params_ = search.best_params_
                    self.best_kparams_ = params
                    self.best_estimator_ = search.best_estimator_

    def predict(self, X):
        Y = self.X
        gram = kf.k0(X, Y, **self.best_kparams_)
        return self.best_estimator_.predict(gram)


class SearchK1(BaseSearch):
    """Finds the best parameters for :meth:`kcat.kernels.functoins.k1`.

    Args:
        alpha: A list of floats.
        functions: A list with tuples of the form ('prev', 'post').
        gamma: A list of float values.
    """

    def __init__(self, alpha, functions, gamma, **kwargs):
        self.alpha = alpha
        self.functions = functions
        self.gamma = gamma
        super().__init__(**kwargs)

    def fit(self, X, y):
        self.X = X
        self.pgen = pgen(X)
        Xp = self.pgen(X)
        # Only 'f1' and 'f2' use gammas, no need to search all the
        # permutations.
        for prev, post in self.functions:
            uses_gammas = prev == 'f1' or post in ('f1', 'f2')
            for g in self.gamma if uses_gammas else [None]:
                for a in self.alpha:
                    search = GridSearchCV(**self.gskwargs)
                    params = dict(alpha=a, prev=prev, post=post, gamma=g)
                    gram = kf.k1(X, X, Xp, Xp, **params)
                    search.fit(gram, y)
                    if search.best_score_ >= self.best_score_:
                        self.best_score_ = search.best_score_
                        self.best_params_ = search.best_params_
                        self.best_kparams_ = params
                        self.best_estimator_ = search.best_estimator_

    def predict(self, X):
        Xp  = self.pgen(X)
        Y = self.X
        Yp = self.pgen(Y)
        gram = kf.k1(X, Y, Xp, Yp, **self.best_kparams_)
        return self.best_estimator_.predict(gram)


class SearchK2(BaseSearch):
    """Finds the best parameters for :meth:`kcat.kernels.functoins.k2`.

    Args:
        functions: A list with tuples of the form ('prev', 'post').
        gamma: A list of float values.
    """

    def __init__(self, functions, gamma, **kwargs):
        self.functions = functions
        self.gamma = gamma
        super().__init__(**kwargs)

    def fit(self, X, y):
        self.X = X
        self.pgen = pgen(X)
        Xp = self.pgen(X)
        # Only 'f1' and 'f2' use gammas, no need to search all the
        # permutations.
        for prev, post in self.functions:
            uses_gammas = prev == 'f1' or post in ('f1', 'f2')
            for g in self.gamma if uses_gammas else [None]:
                search = GridSearchCV(**self.gskwargs)
                params = dict(prev=prev, post=post, gamma=g)
                gram = kf.k2(X, X, Xp, Xp, **params)
                search.fit(gram, y)
                if search.best_score_ >= self.best_score_:
                    self.best_score_ = search.best_score_
                    self.best_params_ = search.best_params_
                    self.best_kparams_ = params
                    self.best_estimator_ = search.best_estimator_

    def predict(self, X):
        Xp  = self.pgen(X)
        Y = self.X
        Yp = self.pgen(Y)
        gram = kf.k2(X, Y, Xp, Yp, **self.best_kparams_)
        return self.best_estimator_.predict(gram)


class SearchM1(BaseSearch):
    """Finds the best parameters for :meth:`kcat.kernels.functoins.m1`.

    Args:
        alpha: A list of floats.
        functions: A list with tuples of the form ('prev', 'post').
        gamma: A list of float values.
    """

    def __init__(self, alpha, functions, gamma, **kwargs):
        self.alpha = alpha
        self.functions = functions
        self.gamma = gamma
        super().__init__(**kwargs)

    def fit(self, X, y):
        self.X = X
        self.pgen = pgen(X)
        Xp = self.pgen(X)
        for prev, post in self.functions:
            uses_gammas = prev == 'f1' or post == 'f1'
            for g in self.gamma if uses_gammas else [None]:
                for a in self.alpha:
                    search = GridSearchCV(**self.gskwargs)
                    params = dict(alpha=a, prev=prev, post=post, gamma=g)
                    gram = kf.m1(X, X, Xp, Xp, **params)
                    search.fit(gram, y)
                    if search.best_score_ >= self.best_score_:
                        self.best_score_ = search.best_score_
                        self.best_params_ = search.best_params_
                        self.best_kparams_ = params
                        self.best_estimator_ = search.best_estimator_

    def predict(self, X):
        Xp  = self.pgen(X)
        Y = self.X
        Yp = self.pgen(Y)
        gram = kf.m1(X, Y, Xp, Yp, **self.best_kparams_)
        return self.best_estimator_.predict(gram)


class SearchELK(BaseSearch):
    """Finds the best parameters for :meth:`kcat.kernels.functoins.elk`.
    """

    def fit(self, X, y):
        self.X = X
        search = GridSearchCV(**self.gskwargs)
        gram = kf.elk(X, X)
        search.fit(gram, y)
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        self.best_estimator_ = search.best_estimator_

    def predict(self, X):
        gram = kf.elk(X, self.X)
        return self.best_estimator_.predict(gram)

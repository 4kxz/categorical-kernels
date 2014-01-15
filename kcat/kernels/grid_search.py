"""Classes to use Grid Search on the custom kernels defined in
:mod:`kcat.kernels.functions`.

Their interface is very similar to `GridSearchCV
<http://scikit-learn.org/stable/modules/generated/\
sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV>`_
of scikit-learn, and most parameters from GridSearchCV can be used.

For convenience, default parameters for the search are defined in
:mod:`kcat.kernels.parameters` but normal parameters can be passed.
"""

from sklearn.grid_search import GridSearchCV

from . import functions as fn


class GridSearchWrapper:
    """Base class to extend in custom grid searches.

    Kernel specific keyword arguments (*alpha*, *prev*, *post*, etc.)
    should be defined in the subclasses and the rest (*estimator*, *cv*,
    etc.) passed down to this class.
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

    def fit(self, X, y):
        """Fit the model to the data matrix *X* and class vector *y*.

        This method works with the default GridSearchCV.
        Subclasses override this method to peform custom searches.
        """
        result = GridSearchCV(**self.gskwargs)
        result.fit(X, y)
        self.best_estimator_ = result.best_estimator_
        self.best_params_ = result.best_params_
        self.best_score_ = result.best_score_

    @property
    def details(self):
        """A dictionary with all the interesting information about
        the estimator found in the search.
        """
        details = {
            'train_score': self.best_score_,
            'best_parameters': {},
            }
        details['best_parameters'].update(self.best_params_)
        details['best_parameters'].update(self.best_kparams_)
        return details


class GridSearchK0(GridSearchWrapper):
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
        # Only 'f1' and 'f2' use gammas, no need to search all the
        # permutations.
        for prev, post in self.functions:
            uses_gammas = prev == 'f1' or post in ('f1', 'f2')
            for g in self.gamma if uses_gammas else [None]:
                result = GridSearchCV(**self.gskwargs)
                params = dict(prev=prev, post=post, gamma=g)
                gram = fn.fast_k0(X, X, **params)
                result.fit(gram, y)
                if result.best_score_ >= self.best_score_:
                    self.best_score_ = result.best_score_
                    self.best_params_ = result.best_params_
                    self.best_kparams_ = params
                    self.best_estimator_ = result.best_estimator_


class GridSearchK1(GridSearchWrapper):
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

        :param pgen: A probability distribution.

        About *pgen*, see :meth:`~kcat.pgen.get_pgen`.
        """
        self.X = X
        # Only 'f1' and 'f2' use gammas, no need to search all the
        # permutations.
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
                        self.best_params_ = result.best_params_
                        self.best_kparams_ = params
                        self.best_estimator_ = result.best_estimator_


class GridSearchK2(GridSearchWrapper):
    """Finds the best parameters for *K2*.

    :param functions: A list with the 'prev' and 'post' functions.
    :param gamma: A list of values."""

    def __init__(self, functions, gamma, **kwargs):
        self.functions = functions
        self.gamma = gamma
        super().__init__(**kwargs)

    def fit(self, X, y, pgen):
        """Fit the model to the data matrix *X* and class vector *y*.
        :param pgen: A probability distribution.

        About *pgen*, see :meth:`~kcat.pgen.get_pgen`.
        """
        self.X = X
        # Only 'f1' and 'f2' use gammas, no need to search all the
        # permutations.
        for prev, post in self.functions:
            uses_gammas = prev == 'f1' or post in ('f1', 'f2')
            for g in self.gamma if uses_gammas else [None]:
                result = GridSearchCV(**self.gskwargs)
                params = dict(prev=prev, post=post, gamma=g)
                gram = fn.fast_k2(X, X, pgen, **params)
                result.fit(gram, y)
                if result.best_score_ >= self.best_score_:
                    self.best_score_ = result.best_score_
                    self.best_params_ = result.best_params_
                    self.best_kparams_ = params
                    self.best_estimator_ = result.best_estimator_


class GridSearchM1(GridSearchWrapper):
    """Finds the best parameters for *M1*.

    :param alpha: A list of values."""

    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super().__init__(**kwargs)

    def fit(self, X, y, pgen):
        """Fit the model to the data matrix *X* and class vector *y*.
        :param pgen: A probability distribution.

        About *pgen*, see :meth:`~kcat.pgen.get_pgen`.
        """
        self.X = X
        for a in self.alpha:
            result = GridSearchCV(**self.gskwargs)
            params = dict(alpha=a)
            gram = fn.fast_k1(X, X, pgen, **params)
            result.fit(gram, y)
            if result.best_score_ >= self.best_score_:
                self.best_score_ = result.best_score_
                self.best_params_ = result.best_params_
                self.best_kparams_ = params
                self.best_estimator_ = result.best_estimator_


class GridSearchELK(GridSearchWrapper):
    """Finds the best parameters for *ELK*."""

    def fit(self, X, y):
        """Fit the model to the data matrix *X* and class vector *y*."""
        self.X = X
        result = GridSearchCV(**self.gskwargs)
        gram = fn.elk(X, X)
        result.fit(gram, y)
        self.best_params_ = result.best_params_
        self.best_score_ = result.best_score_
        self.best_estimator_ = result.best_estimator_

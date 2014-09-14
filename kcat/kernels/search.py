"""Classes to perform GridSearch on the custom kernels defined in
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
    """The default `GridSearchCV` in scikit-learn searches all possible
    combinations of parameters. With some kernels this is not necessary
    as some combinations of parameters do not make sense
    (eg: prev='f1' with post='f1').

    BaseSearch is a class that can be extended to search an arbitrary
    parameter space, instead of all the possible ones. This is done
    by implementing the function `fit`. Any subclass should deal with
    kernel-specific keyword arguments such as *alpha*, *gamma*, *prev*,
    etc. Common arguments like *estimator*, *cv*, *C* and so on can be
    handled by `GridSearchCV`.
    """
    kernel_function = None

    def __init__(self, estimator, cv, **kwargs):
        self.gskwargs = {
            'estimator': estimator,
            'cv': cv,
            'param_grid': kwargs,
            'n_jobs': 4,
            }
        self.best_score_ = 0
        self.best_params_ = {}
        self.best_kparams_ = {}
        self.best_estimator_ = None
        self.X = None

    def fit(self, X, y):
        """Fit the model to the data matrix *X* and class vector *y*.

        Args:
            X: Numpy matrix with the examples in rows.
            y: Numpy array with the class of each example.
        """
        self.X = X
        G = self.kernel(X, X)
        search = GridSearchCV(**self.gskwargs)
        search.fit(G, y)
        self.best_estimator_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        if search.best_score_ >= self.best_score_:
            self.best_params_ = search.best_params_
            self.best_score_ = search.best_score_
            self.best_estimator_ = search.best_estimator_

    def predict(self, X):
        """
        Args:
            X: Numpy matrix with the examples in rows.

        Returns:
            A Numpy vector with the predicted classes.
        """
        if self.X is None:
            raise ValueError("Model is not fitted.")
        G = self.kernel(X, self.X)
        return self.best_estimator_.predict(G)

    @classmethod
    def kernel(cls, *args, **kwargs):
        """Calls the kernel function associated with the current class."""
        if cls.kernel_function is None:
            return args[0]
        else:
            return cls.kernel_function(*args, **kwargs)

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


class ELKSearch(BaseSearch):
    """Finds the best parameters for :meth:`kcat.kernels.functions.elk`."""
    kernel_function = kf.elk


class K0Search(BaseSearch):
    """Finds the best parameters for :meth:`kcat.kernels.functions.k0`.

    Args:
        functions: A list with tuples of the form ('prev', 'post').
        gamma: A list of floats with the gamma values.
    """
    kernel_function = kf.k0

    def __init__(self, functions, gamma, **kwargs):
        self.functions = functions
        self.gamma = gamma
        super().__init__(**kwargs)

    def fit(self, X, y):
        self.X = X
        for prev, post in self.functions:
            uses_gammas = prev == 'f1' or post in ('f1', 'f2')
            for g in self.gamma if uses_gammas else [None]:
                search = GridSearchCV(**self.gskwargs)
                params = dict(prev=prev, post=post, gamma=g)
                gram = self.kernel(X, X, **params)
                search.fit(gram, y)
                if search.best_score_ >= self.best_score_:
                    self.best_score_ = search.best_score_
                    self.best_params_ = search.best_params_
                    self.best_kparams_ = params
                    self.best_estimator_ = search.best_estimator_

    def predict(self, X):
        Y = self.X
        gram = self.kernel(X, Y, **self.best_kparams_)
        return self.best_estimator_.predict(gram)


class K1Search(BaseSearch):
    """Finds the best parameters for :meth:`kcat.kernels.functions.k1`.

    Args:
        alpha: A list of floats.
        functions: A list with tuples of the form ('prev', 'post').
        gamma: A list of float values.
    """
    kernel_function = kf.k1

    def __init__(self, alpha, functions, gamma, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.functions = functions
        self.gamma = gamma
        self.pgen = None

    def fit(self, X, y):
        self.X = X
        self.pgen = pgen(X)
        self.Xp = Xp = self.pgen(X)
        for prev, post in self.functions:
            uses_gammas = prev == 'f1' or post in ('f1', 'f2')
            for g in self.gamma if uses_gammas else [None]:
                for a in self.alpha:
                    search = GridSearchCV(**self.gskwargs)
                    params = dict(alpha=a, prev=prev, post=post, gamma=g)
                    gram = self.kernel(X, X, Xp, Xp, **params)
                    search.fit(gram, y)
                    if search.best_score_ >= self.best_score_:
                        self.best_score_ = search.best_score_
                        self.best_params_ = search.best_params_
                        self.best_kparams_ = params
                        self.best_estimator_ = search.best_estimator_

    def predict(self, X):
        Xp  = self.pgen(X)
        gram = self.kernel(X, self.X, Xp, self.Xp, **self.best_kparams_)
        return self.best_estimator_.predict(gram)


class K2Search(BaseSearch):
    """Finds the best parameters for :meth:`kcat.kernels.functions.k2`.

    Args:
        functions: A list with tuples of the form ('prev', 'post').
        gamma: A list of float values.
    """
    kernel_function = kf.k2

    def __init__(self, functions, gamma, **kwargs):
        super().__init__(**kwargs)
        self.functions = functions
        self.gamma = gamma
        self.pgen = None

    def fit(self, X, y):
        self.X = X
        self.pgen = pgen(X)
        self.Xp = Xp = self.pgen(X)
        for prev, post in self.functions:
            uses_gammas = prev == 'f1' or post in ('f1', 'f2')
            for g in self.gamma if uses_gammas else [None]:
                search = GridSearchCV(**self.gskwargs)
                params = dict(prev=prev, post=post, gamma=g)
                gram = self.kernel(X, X, Xp, Xp, **params)
                search.fit(gram, y)
                if search.best_score_ >= self.best_score_:
                    self.best_score_ = search.best_score_
                    self.best_params_ = search.best_params_
                    self.best_kparams_ = params
                    self.best_estimator_ = search.best_estimator_

    def predict(self, X):
        Xp  = self.pgen(X)
        gram = self.kernel(X, self.X, Xp, self.Xp, **self.best_kparams_)
        return self.best_estimator_.predict(gram)


class M3Search(K1Search):
    """Finds the best parameters for :meth:`kcat.kernels.functions.m3`.

    Args:
        alpha: A list of floats.
        functions: A list with tuples of the form ('prev', 'post').
        gamma: A list of float values.
    """
    kernel_function = kf.m3


class M4Search(K1Search):
    kernel_function = kf.m4


class M5Search(K1Search):
    kernel_function = kf.m5


class M6Search(K1Search):
    kernel_function = kf.m6


class M7Search(K1Search):
    kernel_function = kf.m7


class M8Search(K1Search):
    kernel_function = kf.m8


class M9Search(K1Search):
    kernel_function = kf.m9


class MASearch(K1Search):
    kernel_function = kf.mA


class MBSearch(K1Search):
    kernel_function = kf.mB


class MCSearch(K1Search):
    kernel_function = kf.mC


class MDSearch(K1Search):
    kernel_function = kf.mD


class MESearch(K1Search):
    kernel_function = kf.mE


class RBFSearch(BaseSearch):
    pass


class Chi1Search(BaseSearch):
    kernel_function = kf.chi1


class Chi2Search(BaseSearch):
    kernel_function = kf.chi2

from sklearn.grid_search import GridSearchCV

from .functions import fast_k0, fast_k1, fast_k2


class GridSearchK0:

    def __init__(self, clf, functions, gammas, **kwargs):
        self.clf = clf
        self.functions = functions
        self.gammas = gammas
        self.params = kwargs
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = 0

    def fit(self, X, y):
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

    def __init__(self, clf, **kwargs):
        self.clf = clf
        self.params = kwargs
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = 0

    def fit(self, X, y, pgen):
        gram = fast_k2(X, X, pgen)
        result = GridSearchCV(self.clf, **self.params)
        result.fit(gram, y)
        self.best_estimator_ = result.best_estimator_
        self.best_params_ = result.best_params_
        self.best_score_ = result.best_score_

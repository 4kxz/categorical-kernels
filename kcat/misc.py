import collections

import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

from .kernels.categorical import fast_k0, fast_k1



def preprocess(X):
    """
    Takes a matrix with a categorical variables and encodes them
    using ¿binarization? Say, for three categories:

    * 0 → 1 0 0
    * 1 → 0 1 0
    * 2 → 0 0 1

    Whatever that's called. So that it can be used with RBF kernel.
    """
    n, d = X.shape
    y = [[] for _ in range(n)]
    # For each attribute:
    for i in range(d):
        # Get the set of values from the examples:
        values = set()
        for j in range(n):
            values.add(X[j][i])
        # Assign an index to each value:
        index = dict()
        for j, v in enumerate(values):
            index[v] = j
        # Assing len(values) attributes to the example where
        # all but the value of the current one are 0
        for j in range(n):
            v = X[j][i]
            new = [0] * len(values)
            new[index[v]] = 1
            y[j] += new
    return np.array(y)


Results = collections.namedtuple('Results', 'best_score_, best_params_')


class GridSearchK0:

    def __init__(self, functions, gammas, **kwargs):
        self.functions = functions
        self.gammas = gammas
        self.kwargs = kwargs

    def fit(self, X, y):
        clf = SVC(kernel='precomputed')
        best_result = Results(best_score_=0, best_params_=None)
        for prev, comp, post in self.functions:
            uses_gammas = prev == 'f1' or post in ('f1', 'f2')
            for g in self.gammas if uses_gammas else [None]:
                p = dict(prev=prev, comp=comp, post=post, gamma=g)
                gram = fast_k0(X, X, **p)
                grid = GridSearchCV(clf, **self.kwargs)
                grid.fit(gram, y)
                if best_result.best_score_ <= grid.best_score_:
                    params = grid.best_params_, p
                    best_result = Results(grid.best_score_, params)
        return best_result


class GridSearchK1:

    def __init__(self, alphas, functions, gammas, **kwargs):
        self.alphas = alphas
        self.functions = functions
        self.gammas = gammas
        self.kwargs = kwargs

    def fit(self, X, X_pgen, y):
        clf = SVC(kernel='precomputed')
        best_result = Results(best_score_=0, best_params_=None)
        for prev, comp, post in self.functions:
            uses_gammas = prev == 'f1' or post in ('f1', 'f2')
            for g in self.gammas if uses_gammas else [None]:
                for a in self.alphas:
                    p = dict(alpha=a, prev=prev, comp=comp, post=post, gamma=g)
                    gram = fast_k1(X, X, X_pgen, **p)
                    grid = GridSearchCV(clf, **self.kwargs)
                    grid.fit(gram, y)
                    if best_result.best_score_ <= grid.best_score_:
                        params = grid.best_params_, p
                        best_result = Results(grid.best_score_, params)
        return best_result


if __name__ == '__main__':
    x = np.array([['a', 'A'], ['b', 'B'], ['c', 'A']])
    print("Categorical")
    print(x)
    print("Binary")
    print(preprocess(x))

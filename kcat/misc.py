import numpy as np


def binary_encoder(X):
    """Takes a matrix with a categorical variables and encodes them in dummy
    variable form. Say, for three categories:

    * 0 → 1 0 0
    * 1 → 0 1 0
    * 2 → 0 0 1

    So that it can be used with RBF kernel.
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

import numpy as np


def preprocess(x):
    """
    Takes a matrix with a categorical variables and encodes them
    using ¿binarization? Say, for three categories:
    0 -> 1 0 0
    1 -> 0 1 0
    2 -> 0 0 1
    Whatever that's called. So that it can be used with RBF kernel.
    """
    n, d = x.shape
    y = [[] for _ in range(n)]
    # For each attribute:
    for i in range(d):
        # Get the set of values from the examples:
        values = set()
        for j in range(n):
            values.add(x[j][i])
        # Assign an index to each value:
        index = dict()
        for j, v in enumerate(values):
            index[v] = j
        # Assing len(values) attributes to the example where
        # all but the value of the current one are 0
        for j in range(n):
            v = x[j][i]
            new = [0] * len(values)
            new[index[v]] = 1
            y[j] += new
    return np.array(y)


if __name__ == '__main__':
    x = np.array([['a', 'A'], ['b', 'B'], ['c', 'A']])
    print("Categorical")
    print(x)
    print("Binary")
    print(preprocess(x))

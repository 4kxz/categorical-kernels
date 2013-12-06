import collections

import numpy as np


def get_pgen(X):
    """
    Obtains a probability mass function generator from `X`.

    :param X: Matrix where each row is an example and each column a categorical
        attribute.

    :returns: Probability mass function generator :math:`pgen_X`, such that
        ``pgen_x(i)`` returns the function :math:`p_i`. In turn, ``p_i(x)``
        returns the probability of category *x* in the *i*-th attribute.
    """
    m, n = X.shape
    # defaultdict initialises entries to 0.
    pmf = [collections.defaultdict(int) for _ in range(n)]
    for i in range(m):
        for j in range(n):
            c = X[i][j]
            pmf[j][c] += 1.0 / m
    # `pmf` is a list of dict (array<map<symbol, real>>), using it in the
    # kernels would require to pass the indices all the way down to the
    # univarate kernel. Using a couple of lambdas it can be turned into a
    # function generator such that pgen(j) returns the function p(c) for the
    # j-th attribute in pmf, which avoids some clutter in the code.
    return lambda j: lambda c: pmf[j][c]

def apply_pgen(pgen, X):
    """
    Applies `pgen` to each element in `X`.

    :param X: Matrix where each row is an example and each column a categorical
        attribute.

    :returns: Matrix *Y* of size :math:`m \\times n`, where
        :math:`Y_{i, j} = P_j(X_{i, j})`.
    """
    m, n = X.shape
    P = np.zeros(X.shape)
    for i in range(m):
        for j in range(n):
            P[i][j] = pgen(j)(X[i][j])
    return P

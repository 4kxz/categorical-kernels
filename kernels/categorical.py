import collections

import numpy as np


#------------------------------------------------------------------------------
# Misc. Functions
#------------------------------------------------------------------------------

def mean(x):
    return sum(x) / len(x)

def prod(x):
    p = 1
    for xi in x:
        p *= xi
    return p

def f1gen(gamma):
    return lambda k, x, y, *args: np.exp(gamma *  k(x, y, *args))

def f2gen(gamma):
    return lambda k, x, y, *args: np.exp(
        gamma * (2 * k(x, y, *args) - k(x, x, *args) - k(y, y, *args))
    )

def ident(k, *args):
    return k(*args)

def pmf_from_matrix(X):
    """
    `X` is a matrix where each row is an example and each column a categorical
    attribute.

    Returns an array of arrays where ``pmf[j][c]`` is the probability of
    occurence of category ``c = X[i][j]`` within the ``j``-th attribute.`The
    returned structure is a `list` of `dict`.
    """
    n, d = X.shape
    pmf = []
    for j in range(d):
        pmf.append(collections.defaultdict(int))  # Initialises entries to 0.
        for i in range(n):
            c = X[i][j]
            pmf[j][c] += 1 / n
    return pmf

# pmf is an array<map<symbol, real>> but to use it would require to pass the
# indices all the way down to the kernel.
# Using a couple of lambdas it can be turned into a function generator such
# that pgen(i) returns the function p(c) for the i-th attribute in pmf.

def pmf_to_pgen(pmf):
    return lambda i: lambda c: pmf[i][c]


#------------------------------------------------------------------------------
# Categorical Kernel K0
#------------------------------------------------------------------------------

def k0_univ(x, y):
    """Univariate kernel k0 between two single variables."""
    return 0 if x != y else 1

def k0_mult(u, v, prev, comp):
    """
    Multivariate kernel between two vectors `u` and `v` that applies the kernel
    k0 for each attribute and a composition function to return a single value.

    * `prev` is a function to transform the data before applying `comp`.
    * `comp` is a function that takes a vector and returns a single value.
    """
    # List comprehension takes care of applying k0 and prev for each element:
    return comp([prev(k0_univ, u[i], v[i]) for i in range(len(u))])

def k0(X, Y, prev, comp, post):
    """
    `X` and `Y` are both matrices where each row is an example and each column
    a categorical attribute.

    Returns the gram matrix obtained applying ``k0_mult(x, y, prev, comp)``
    between each pair of elements in `X` and `Y`.

    * `prev` is a function to transform the data before applying `comp`.
    * `comp` is a function that takes a vector and returns a single value.
    * `post` is a function to transform the data after applying `comp`.
    """
    # The gram matrix is computed by iterating each vector in X and Y:
    G = np.zeros((len(X), len(Y)))
    for i, u in enumerate(X):
        for j, v in enumerate(Y):
            G[i][j] = post(k0_mult, u, v, prev, comp)
    return G

def fast_k0(X, prev='ident', comp='mean', post='ident', params=None):
    """
    This is an optimised version of `k0`, to be used when *X = Y*.

    It only works with a defined set of functions:

    * `prev` accepts ``'ident'`` and ``'f1'``.
    * `comp` accepts ``'mean'`` and ``'prod'``.
    * `post` accepts ``'ident'``, ``'f1'`` and  ``'f2'``.
    * `params` is a dictionary with any parameter used by the functions.
    """
    # Previous transformation function:
    if prev == 'ident':
        prevf = lambda x: x
    elif prev == 'f1':
        prevf = lambda x: np.exp(params['gamma'] * x)
    else:
        raise ValueError("Unknown previous function {}.".format(prev))
    # Composition function:
    if comp == 'mean':
        compf = lambda x: np.mean(x, axis=1)
    elif comp == 'prod':
        compf = lambda x: np.product(x, axis=1)
    else:
        raise ValueError("Unknown composition function '{}'.".format(comp))
    # Posterior transformation function:
    if post == 'ident':
        postf = lambda x: x
    elif post == 'f1':
        postf = lambda x: np.exp(params['gamma'] * x)
    elif post == 'f2':
        # Since the multivariate kernel is the overlap, k(u, u) is always 1
        # and it can be simplified, independently of whether the composition is
        # the mean or the product:
        postf = lambda x: np.exp(params['gamma'] * (2 * x - 2))
    else:
        raise ValueError("Unknown posterior function {}.".format(post))
    n, d = X.shape
    G = np.zeros((n, n))
    # The gram matrix is computed using vectorised operations because speed:
    for i, xi in enumerate(X):
        Xi = np.repeat([xi], n - i, axis=0)
        Gi = X[i:n] == Xi
        Gi = postf(compf(prevf(Gi)))
        G[i, i:n] = G[i:n, i] = Gi
    return G


#------------------------------------------------------------------------------
# Categorical Kernel K1
#------------------------------------------------------------------------------

def k1_univ(x, y, h, p):
    """Univariate kernel k1 between two single variables."""
    return 0 if x != y else h(p(x))

def k1_mult(u, v, prev, comp, h, pgen):
    """
    Multivariate kernel between two vectors `u` and `v` that applies the kernel
    k0 for each attribute and a composition function to return a single value.

    * `prev` is a function to transform the data before applying `comp`.
    * `comp` is a function that takes a vector and returns a single value.
    * `h` is the inverting function.
    * `pgen` is a probability function generator (*see pmf_to_pgen*).
    """
    # Compute the kernel applying the previous transformation and composition functions:
    return comp([prev(k1_univ, u[i], v[i], h, pgen(i)) for i in range(len(u))])

def k1(X, Y, prev, comp, post, alpha=1, pmf=None):
    """
    `X` and `Y` are both matrices where each row is an example and each column
    a categorical attribute.

    Returns the gram matrix obtained applying ``k1_mult(x, y, prev, comp)``
    between each pair of elements in `X` and `Y`.

    * `prev` is a function to transform the data before applying `comp`.
    * `comp` is a function that takes a vector and returns a single value.
    * `post` is a function to transform the data after applying `comp`.
    * `alpha` is the parameter for the inverting function *h*.
    * `pmf` is the probability mass function (*by default pmf_from_matrix*).
    """
    # When pmf is unknown compute it from X:
    if pmf is None:
        pmf = pmf_from_matrix(X)
    pgen = pmf_to_pgen(pmf)
    # Inverting function h_a:
    h = lambda x: (1 - x ** alpha) ** (1 / alpha)
    # Compute the kernel matrix:
    M = np.zeros((len(X), len(Y)))
    for i, u in enumerate(X):
        for j, v in enumerate(Y):
            M[i][j] = post(k1_mult, u, v, prev, comp, h, pgen)
    return M

def fast_k1(X, prev='ident', comp='mean', post='ident', params=None, pmf=None):
    """
    This is an optimised version of `k1`, to be used when *X = Y*.

    It only works with a defined set of functions:

    * `prev` accepts ``'ident'`` and ``'f1'``.
    * `comp` accepts ``'mean'`` and ``'prod'``.
    * `post` accepts ``'ident'``, ``'f1'`` and  ``'f2'``.
    * `params` is a dictionary with any parameter used by the functions.
    * `pmf` is the probability mass function (*by default pmf_from_matrix*).
    """
    if params is None:
        params = {'alpha': 1}
    if pmf is None:
        pmf = pmf_from_matrix(X)
    # Inverting function h:
    h = lambda x: (1 - x ** params['alpha']) ** (1 / params['alpha'])
    # Previous transformation function:
    if prev == 'ident':
        prevf = lambda x: x
    elif prev == 'f1':
        prevf = lambda x: np.exp(params['gamma'] * x)
    else:
        raise ValueError("Unknown previous function {}.".format(prev))
    # Composition function:
    if comp == 'mean':
        compf = lambda x: np.mean(x, axis=1)
    elif comp == 'prod':
        compf = lambda x: np.product(x, axis=1)
    else:
        raise ValueError("Unknown composition function '{}'.".format(comp))
    # Posterior transformation function:
    if post == 'ident':
        postf = lambda x: x
    elif post == 'f1':
        postf = lambda x: np.exp(params['gamma'] * x)
    elif post == 'f2':
        postf = lambda x: x  # The post f2 is applied later in a seperate loop.
    else:
        raise ValueError("Unknown posterior function {}.".format(post))
    # Create a matrix with the weights, for convenience:
    n, d = X.shape
    P = np.zeros(X.shape)
    for i in range(n):
        for j in range(d):
            v = X[i][j]
            P[i][j] = h(pmf[j][v])  # Apply h(x) to all the categories in pmf.
    # The gram matrix is computed using vectorised operations because speed:
    G = np.zeros((n, n))
    for i in range(n):
        Xi = np.repeat([X[i]], n - i, axis=0)
        Pi = np.repeat([P[i]], n - i, axis=0)
        Gi = (X[i:n] == Xi) * Pi
        Gi = postf(compf(prevf(Gi)))
        G[i, i:n] = G[i:n, i] = Gi
    # When post == 'f2', postf does nothing.
    # The actual post function is applied here:
    if post == 'f2':
        # We know that: f2 = e ^ (gamma * (2 * k(x, y) - k(x, x) - k(y, y))).
        # The current values of G are those of k(x, y).
        # The values of k(z, z) are computed in the diagonal of G for any z.
        # We can use that to avoid recomputing k(x, x) and k(y, y):
        kxx = np.diag(G)
        kyy = kxx[np.newaxis].T
        G = np.exp(params['gamma'] * (2 * G - kxx - kyy))
    return G


#------------------------------------------------------------------------------
# Categorical Kernel K2
#------------------------------------------------------------------------------

def fast_k2(X, pmf=None):
    """
    This is an optimised version of `k2`, to be used when *X = Y*.

    * `pmf` is the probability mass function (*by default pmf_from_matrix*).
    """
    if pmf is None:
        pmf = pmf_from_matrix(X)
    # Create a matrix with the weights, for convenience:
    n, d = X.shape
    P = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            v = X[i][j]
            P[i][j] = pmf[j][v]
    # The gram matrix is computed using vectorised operations because speed:
    M = np.zeros((n, n))
    for i in range(n):
        Xi = np.repeat([X[i]], n - i, axis=0)
        Pi = np.repeat([P[i]], n - i, axis=0)
        Pi = (1 / Pi) / n
        Mi = (X[i:n] == Xi) * Pi
        Mi = Mi.sum(axis=1) / d
        M[i, i:n] = M[i:n, i] = Mi
    return M

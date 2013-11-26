import collections

import numpy as np


#------------------------------------------------------------------------------
# Misc. Functions
#------------------------------------------------------------------------------

def get_function(name, params=None):
    if callable(name):
        return name
    elif name == 'mean':
        return lambda x: np.mean(x)
    elif name == 'prod':
        return lambda x: np.prod(x)
    elif name == 'ident':
        return lambda k, *a: k(*a)
    elif name == 'f1':
        return lambda k, x, y, *a: np.exp(
            params['gamma'] * k(x, y, *a)
        )
    elif name == 'f2':
        return lambda k, x, y, *a: np.exp(
            params['gamma'] * (2.0 * k(x, y, *a) - k(x, x, *a) - k(y, y, *a))
        )
    else:
        raise ValueError("Invalid function")

def get_vectorised_function(name, params=None):
    if callable(name):
        return name
    elif name == 'mean':
        return lambda x: np.mean(x, axis=1)
    elif name == 'prod':
        return lambda x: np.prod(x, axis=1)
    elif name == 'ident':
        return lambda x: x
    elif name == 'f1':
        return lambda x: np.exp(params['gamma'] * x)
    else:
        raise ValueError("Invalid function")

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
            pmf[j][c] += 1.0 / n
    return pmf

# pmf is an array<map<symbol, real>> but to use it would require to pass the
# indices all the way down to the kernel.
# Using a couple of lambdas it can be turned into a function generator such
# that pg(i) returns the function p(c) for the i-th attribute in pmf.

def pmf_to_pg(pmf):
    return lambda i: lambda c: pmf[i][c]


#------------------------------------------------------------------------------
# Categorical Kernel K0
#------------------------------------------------------------------------------

def k0_univ(x, y):
    """Univariate kernel k0 between two single variables."""
    return 0.0 if x != y else 1.0

def k0_mult(u, v, prevf, compf):
    """
    Multivariate kernel between two vectors `u` and `v` that applies the kernel
    k0 for each attribute and a composition function to return a single value.

    * `prevf` is a function to transform the data before applying `comp`.
    * `compf` is a function that takes a vector and returns a single value.
    """
    # List comprehension takes care of applying k0 and prev for each element:
    return compf([prevf(k0_univ, u[i], v[i]) for i in range(len(u))])

def k0(X, Y, prev='ident', comp='mean', post='ident', params=None):
    """
    `X` and `Y` are both matrices where each row is an example and each column
    a categorical attribute.

    Returns the gram matrix obtained applying ``k0_mult(x, y, prev, comp)``
    between each pair of elements in `X` and `Y`.

    * `prev` is a function to transform the data before applying `comp`.
    * `comp` is a function that takes a vector and returns a single value.
    * `post` is a function to transform the data after applying `comp`.
    """
    prevf = get_function(prev, params)
    compf = get_function(comp, params)
    postf =  get_function(post, params)
    # The gram matrix is computed by iterating each vector in X and Y:
    G = np.zeros((len(X), len(Y)))
    for i, u in enumerate(X):
        for j, v in enumerate(Y):
            G[i][j] = postf(k0_mult, u, v, prevf, compf)
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
    # Since the multivariate kernel is the overlap, k(u, u) is always 1
    # and it can be simplified, independently of whether the composition is
    # the mean or the product:
    if post == 'f2':
        post = lambda x: np.exp(params['gamma'] * (2.0 * x - 2.0))
    prevf = get_vectorised_function(prev, params)
    compf = get_vectorised_function(comp, params)
    postf =  get_vectorised_function(post, params)
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
    return 0.0 if x != y else h(p(x))

def k1_mult(u, v, prevf, compf, h, pg):
    """
    Multivariate kernel between two vectors `u` and `v` that applies the kernel
    k1 for each attribute and a composition function to return a single value.

    * `prevf` is a function to transform the data before applying `comp`.
    * `compf` is a function that takes a vector and returns a single value.
    * `h` is the inverting function.
    * `pg` is a probability function generator (*see pmf_to_pg*).
    """
    # Compute the kernel applying the previous and composition functions:
    return compf([prevf(k1_univ, u[i], v[i], h, pg(i)) for i in range(len(u))])

def k1(X, Y, prev='ident', comp='mean', post='ident', params=None, pmf=None):
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
    prevf = get_function(prev, params)
    compf = get_function(comp, params)
    postf =  get_function(post, params)
    # When pmf is unknown compute it from X:
    if pmf is None:
        pmf = pmf_from_matrix(X)
    pg = pmf_to_pg(pmf)
    # Inverting function h_a:
    alpha = 1.0 if params is None else params.get('alpha', 1.0)
    h = lambda x: (1.0 - x ** alpha) ** (1.0 / alpha)
    # Compute the kernel matrix:
    G = np.zeros((len(X), len(Y)))
    for i, u in enumerate(X):
        for j, v in enumerate(Y):
            G[i][j] = postf(k1_mult, u, v, prevf, compf, h, pg)
    return G

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
    if pmf is None:
        pmf = pmf_from_matrix(X)
    # Inverting function h:
    alpha = 1.0 if params is None else params.get('alpha', 1.0)
    h = lambda x: (1.0 - x ** alpha) ** (1.0 / alpha)
    # The post f2 is applied later in a seperate loop.
    if post == 'f2':
        post = lambda x: x
    prevf = get_vectorised_function(prev, params)
    compf = get_vectorised_function(comp, params)
    postf =  get_vectorised_function(post, params)
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
        G = np.exp(params['gamma'] * (2.0 * G - kxx - kyy))
    return G


#------------------------------------------------------------------------------
# Categorical Kernel K2
#------------------------------------------------------------------------------

def k2_univ(x, y, p, n):
    """Univariate kernel k2 between two single variables."""
    return 0.0 if x != y else 1.0 / p(x) / n

def k2_mult(u, v, pg, n):
    """
    Multivariate kernel between two vectors `u` and `v` that applies the kernel
    k2 for each attribute and a composition function to return a single value.

    * `pg` is a probability function generator (*see pmf_to_pg*).
    """
    # Compute the kernel applying the previous and composition functions:
    return np.mean([k2_univ(u[i], v[i], pg(i), n) for i in range(len(u))])

def k2(X, Y, pmf=None):
    """
    `X` and `Y` are both matrices where each row is an example and each column
    a categorical attribute.

    Returns the gram matrix obtained applying ``k2_mult(x, y, prev, comp)``
    between each pair of elements in `X` and `Y`.

    * `pmf` is the probability mass function (*by default pmf_from_matrix*).
    """
    # When pmf is unknown compute it from Y:
    if pmf is None:
        pmf = pmf_from_matrix(Y)
    pg = pmf_to_pg(pmf)
    # Compute the kernel matrix:
    G = np.zeros((len(X), len(Y)))
    for i, u in enumerate(X):
        for j, v in enumerate(Y):
            G[i][j] = k2_mult(u, v, pg, len(Y))
    return G

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
    G = np.zeros((n, n))
    for i in range(n):
        Xi = np.repeat([X[i]], n - i, axis=0)
        Pi = np.repeat([P[i]], n - i, axis=0)
        Pi = 1.0 / Pi / n
        Gi = (X[i:n] == Xi) * Pi
        Gi = Gi.sum(axis=1) / d
        G[i, i:n] = G[i:n, i] = Gi
    return G

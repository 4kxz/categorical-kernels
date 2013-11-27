import collections

import numpy as np


#------------------------------------------------------------------------------
# Misc. Functions
#------------------------------------------------------------------------------

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
    # TODO
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

# Some of the categorical kernels can receive Python functions as parameters.
# For ease of use, some predefined functions can be specified with a string.
# `get_function` and `get_vector_function` take this string and return the
# appropiate function. Both are used when handling the kernel parameters.

def get_function(name, params={}):
    if callable(name):
        return name
    elif name == 'mean':
        return lambda x: np.mean(x)
    elif name == 'prod':
        return lambda x: np.prod(x)
    elif name == 'ident':
        return lambda k, *args, **kwargs: k(*args, **kwargs)
    elif name == 'f1':
        def f1(k, x, y, *args, **kwargs):
            kxy = k(x, y, *args, **kwargs)
            return np.exp(params['gamma'] * kxy)
        return f1
    elif name == 'f2':
        def f2(k, x, y, *args, **kwargs):
            kxy = k(x, y, *args, **kwargs)
            kxx = k(x, x, *args, **kwargs)
            kyy = k(y, y, *args, **kwargs)
            return np.exp(params['gamma'] * (2.0 * kxy - kxx - kyy))
        return f2
    else:
        raise ValueError("Invalid function {}".format(name))

def get_vector_function(name, params={}):
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
    elif name == 'f2':
        raise ValueError("Function f2 can't be vectorised")
    else:
        raise ValueError("Invalid function {}".format(name))


#------------------------------------------------------------------------------
# Categorical Kernel K0
#------------------------------------------------------------------------------

def k0_univ(x, y):
    """Univariate kernel k0."""
    return 0.0 if x != y else 1.0

def k0_mult(u, v, prev, comp):
    """
    Multivariate kernel k0.

    :param u: Data vector.
    :param v: Data vector.
    :param prev: Function to transform the data before applying `comp`.
    :param comp: Function that takes a vector and returns a single value.

    :returns: Value of applying the kernel ``k0_univ`` between each pair of
        attributes in `u` and `v`, and then the composition function.
    """
    # List comprehension takes care of applying k0 and prev for each element:
    return comp([prev(k0_univ, u[i], v[i]) for i in range(len(u))])

def k0(X, Y, prev='ident', comp='mean', post='ident', **kwargs):
    """
    Computes the gram matrix.

    :param X: Data matrix where each row is an example and each column a
        categorical attribute.
    :param Y: Data matrix.
    :param prev: Function to transform the data before applying `comp`. Accepts
        ``'ident'`` and ``'f1'`` or a Python function.
    :param comp: Function that takes a vector and returns a single value.
        Accepts ``'mean'`` and ``'prod'`` or a Python function.
    :param post: Function to transform the data after applying `comp`. Accepts
        ``'ident'``, ``'f1'`` and  ``'f2'`` or a Python function.
    :param gamma: (optional) Parameter required by ``'f1'`` and  ``'f2'``.

    :returns: Gram matrix obtained applying ``k0_mult`` between each pair of
        elements in `X` and `Y`.
    """
    prevf = get_function(prev, kwargs)
    compf = get_function(comp, kwargs)
    postf =  get_function(post, kwargs)
    # The gram matrix is computed by iterating each vector in X and Y:
    G = np.zeros((len(X), len(Y)))
    for i, u in enumerate(X):
        for j, v in enumerate(Y):
            G[i][j] = postf(k0_mult, u, v, prevf, compf)
    return G

def fast_k0(X, Y, prev='ident', comp='mean', post='ident', **kwargs):
    """
    An optimised version of *k0* with the same interface.

    Since the code is vectorised any Python functions passed as argument must
    work with numpy arrays.
    """
    # Since the multivariate kernel is the overlap, k(u, u) is always 1
    # and it can be simplified, independently of whether the composition is
    # the mean or the product:
    if post == 'f2':
        gamma = kwargs['gamma']
        post = lambda x: np.exp(gamma * (2.0 * x - 2.0))
    prevf = get_vector_function(prev, kwargs)
    compf = get_vector_function(comp, kwargs)
    postf = get_vector_function(post, kwargs)
    xn, xd = X.shape
    yn, yd = Y.shape
    # The gram matrix is computed using vectorised operations because speed:
    XL = np.repeat(X, yn, axis=0)
    YL = np.tile(Y, (xn, 1))
    G = XL == YL
    G = postf(compf(prevf(G)))
    return G.reshape(xn, yn)

#------------------------------------------------------------------------------
# Categorical Kernel K1
#------------------------------------------------------------------------------

def k1_univ(x, y, h, p):
    """
    Univariate kernel k1.

    :param x: Value.
    :param y: Value.
    :param h: Inverting function.
    :param p: Probability function.
    """
    return 0.0 if x != y else h(p(x))

def k1_mult(u, v, h, pgen, prev, comp):
    """
    Multivariate kernel k1.

    :param u: Data vector.
    :param v: Data vector.
    :param h: Inverting function.
    :param pgen: Probability mass function generator (*see get_pgen*).
    :param prev: Function to transform the data before applying `comp`.
    :param comp: Function that takes a vector and returns a single value.

    :returns: Value of applying the kernel ``k1_univ`` between each pair of
        attributes in `u` and `v`, and then the composition function.
    """
    # Compute the kernel applying the previous and composition functions:
    return comp([prev(k1_univ, u[i], v[i], h, pgen(i)) for i in range(len(u))])

def k1(X, Y, pgen, alpha=1.0, prev='ident', comp='mean', post='ident',
        **kwargs):
    """
    Computes the gram matrix.

    :param X: Data matrix where each row is an example and each column a
        categorical attribute.
    :param Y: Data matrix.
    :param pgen: Probability mass function generator (*see get_pgen*).
    :param alpha: Parameter for the inverting function *h*.
    :param prev: Function to transform the data before applying `comp`. Accepts
        ``'ident'`` and ``'f1'`` or a Python function.
    :param comp: Function that takes a vector and returns a single value.
        Accepts ``'mean'`` and ``'prod'`` or a Python function.
    :param post: Function to transform the data after applying `comp`. Accepts
        ``'ident'``, ``'f1'`` and  ``'f2'`` or a Python function.
    :param gamma: (optional) Parameter required by ``'f1'`` and  ``'f2'``.

    :returns: Gram matrix obtained applying ``k1_mult`` between each pair of
        elements in `X` and `Y`.
    """
    h = lambda x: (1.0 - x ** alpha) ** (1.0 / alpha)
    prevf = get_function(prev, kwargs)
    compf = get_function(comp, kwargs)
    postf =  get_function(post, kwargs)
    # Compute the kernel matrix:
    G = np.zeros((len(X), len(Y)))
    for i, u in enumerate(X):
        for j, v in enumerate(Y):
            G[i][j] = postf(k1_mult, u, v, h, pgen, prevf, compf)
    return G

def fast_k1(X, Y, pgen, alpha=1.0, prev='ident', comp='mean', post='ident',
        **kwargs):
    """
    An optimised version of *k1* with the same interface.

    Since the code is vectorised any Python functions passed as argument must
    work with numpy arrays.
    """
    h = lambda x: (1.0 - x ** alpha) ** (1.0 / alpha)
    Py = h(apply_pgen(pgen, Y))
    prevf = get_vector_function(prev, kwargs)
    compf = get_vector_function(comp, kwargs)
    postf = get_vector_function(post, kwargs) if post != 'f2' else None
    # The function f2 needs to be treated separately.
    xn, xd = X.shape
    yn, yd = Y.shape
    # The gram matrix is computed using vectorised operations because speed:
    XL = np.repeat(X, yn, axis=0)
    YL = np.tile(Y, (xn, 1))
    PY = np.tile(Py, (xn, 1))
    G = (XL == YL) * PY
    G = compf(prevf(G))
    # When post == 'f2', postf does nothing.
    # The actual post function is applied here:
    if post != 'f2':
        G = postf(G)
    else:
        # We know that: f2 = e ^ (gamma * (2 * k(x, y) - k(x, x) - k(y, y))).
        # The current values of G are those of k(x, y).
        # We need to compute the values of k(x, x) and k(y, y) for each
        # x in X and y in Y:
        Px = h(apply_pgen(pgen, X))
        GX = np.repeat(Px, yn, axis=0)
        GY = np.tile(Py, (xn, 1))
        GX = compf(prevf(GX))
        GY = compf(prevf(GY))
        # Apply f2:
        gamma = kwargs['gamma']
        G = np.exp(gamma * (2.0 * G - GX - GY))
    return G.reshape(xn, yn)

#------------------------------------------------------------------------------
# Categorical Kernel K2
#------------------------------------------------------------------------------

def k2_univ(x, y, p, n):
    """
    Univariate kernel k2.

    :param x: Value.
    :param y: Value.
    :param p: Probability function.
    :param n: Number of elements.
    """
    return 0.0 if x != y else 1.0 / p(x) / n

def k2_mult(u, v, pgen, n):
    """
    Multivariate kernel k2.

    :param u: Data vector.
    :param v: Data vector.
    :param pgen: Probability mass function generator (*see get_pgen*).
    :param n: Number of elements.

    :returns: Value of applying the kernel ``k2_univ`` between each pair of
        attributes in `u` and `v`, and then the composition function.
    """
    # Compute the kernel applying the previous and composition functions:
    return np.mean([k2_univ(u[i], v[i], pgen(i), n) for i in range(len(u))])

def k2(X, Y, pgen):
    """
    Computes the gram matrix.

    :param X: Data matrix where each row is an example and each column a
        categorical attribute.
    :param Y: Data matrix.
    :param pgen: Probability mass function generator (*see get_pgen*).

    :returns: Gram matrix obtained applying ``k2_mult`` between each pair of
        elements in `X` and `Y`.
    """
    # Compute the kernel matrix:
    G = np.zeros((len(X), len(Y)))
    for i, u in enumerate(X):
        for j, v in enumerate(Y):
            G[i][j] = k2_mult(u, v, pgen, len(Y))
    return G

def fast_k2(X, Y, pgen):
    """
    An optimised version of *k2* with the same interface.
    """
    xn, xd = X.shape
    yn, yd = Y.shape
    Yp = np.zeros(Y.shape)
    # Create a matrix with the weights, for convenience:
    Yp = 1.0 / apply_pgen(pgen, Y) / yn
    # The gram matrix is computed using vectorised operations because speed:
    XL = np.repeat(X, yn, axis=0)
    YL = np.tile(Y, (xn, 1))
    YP = np.tile(Yp, (xn, 1))
    G = (XL == YL) * YP
    return G.mean(axis=1).reshape(xn, yn)

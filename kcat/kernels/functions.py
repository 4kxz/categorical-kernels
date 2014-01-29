"""This module contains methods to compute the gram matrix using the kernel
functions *K0*, *K1* and *K2*. The gram matrix can be used to train SVMs using
scikit-learn.

The methods :meth:`fast_k0`, :meth:`fast_k1` and :meth:`fast_k2` are
recommended as they take advantage of numpy's vectorial operations and are
much faster to compute. Their drawback is that they don't accept arbitrary
Python functions as parameters.
"""

import numpy as np

from ..utils import get_pgen, apply_pgen


# Some of the categorical kernels can receive Python functions as parameters.
# For ease of use, some predefined functions can be specified with a string.
# `get_function` and `get_vector_function` take this string and return the
# appropiate function. Both are used when handling the kernel parameters.

def get_function(name, params={}):
    if callable(name):
        return name
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
    elif name == 'ident':
        return lambda x: x
    elif name == 'f1':
        return lambda x: np.exp(params['gamma'] * x)
    elif name == 'f2':
        raise ValueError("Function f2 can't be vectorised")
    else:
        raise ValueError("Invalid function {}".format(name))


# Categorical Kernel K0

def k0_univ(x, y):
    """Univariate kernel *K0*."""
    return 0.0 if x != y else 1.0

def k0_mult(u, v, prev):
    """Multivariate kernel *K0*.

    :param u: Data vector.
    :param v: Data vector.
    :param prev: Function to transform the data before composing.

    :returns: Value of applying the kernel :meth:`k0_univ` between each pair of
        attributes in *u* and *v*, and then the composition function.
    """
    # List comprehension takes care of applying k0 and prev for each element:
    return np.mean([prev(k0_univ, u[i], v[i]) for i in range(len(u))])

def k0(X, Y, prev='ident', post='ident', **kwargs):
    """Computes the gram matrix.

    :param X: Data matrix where each row is an example and each column a
        categorical attribute.
    :param Y: Data matrix.
    :param prev: Function to transform the data before composing. Accepts
        ``'ident'``, ``'f1'`` or a Python function.
    :param post: Function to transform the data after composing. Accepts
        ``'ident'``, ``'f1'`` or a Python function.
    :param gamma: (optional) Parameter required by ``'f1'``.

    :returns: Gram matrix obtained applying :meth:`k0_mult` between each pair
        of elements in *X* and *Y*.
    """
    prevf = get_function(prev, kwargs)
    postf =  get_function(post, kwargs)
    # The gram matrix is computed by iterating each vector in X and Y:
    G = np.zeros((len(X), len(Y)))
    for i, u in enumerate(X):
        for j, v in enumerate(Y):
            G[i][j] = postf(k0_mult, u, v, prevf)
    return G

def fast_k0(X, Y, prev='ident', post='ident', **kwargs):
    """An optimised version of :meth:`k0` with the same interface.

    Since the code is vectorised any Python functions passed as argument must
    work with numpy arrays.
    """
    prevf = get_vector_function(prev, kwargs)
    postf = get_vector_function(post, kwargs)
    xm, xn = X.shape
    ym, yn = Y.shape
    # The gram matrix is computed using vectorised operations because speed:
    XL = np.repeat(X, ym, axis=0)
    YL = np.tile(Y, (xm, 1))
    G = XL == YL
    G = postf(np.mean(prevf(G), axis=1))
    return G.reshape(xm, ym)


# Categorical Kernel K1

def k1_univ(x, y, h, p):
    """Univariate kernel *K1*.

    :param x: Value.
    :param y: Value.
    :param h: Inverting function.
    :param p: Probability function.
    """
    return 0.0 if x != y else h(p(x))

def k1_mult(u, v, h, pgen, prev):
    """Multivariate kernel *K1*.

    :param u: Data vector.
    :param v: Data vector.
    :param h: Inverting function.
    :param pgen: Probability mass function generator (see
        :meth:`~kcat.pgen.get_pgen`).
    :param prev: Function to transform the data before composing.

    :returns: Value of applying the kernel :meth:`k1_univ` between each pair of
        attributes in *u* and *v*, and then the composition function.
    """
    # Compute the kernel applying the previous and composition functions:
    r = np.mean([prev(k1_univ, u[i], v[i], h, pgen(i)) for i in range(len(u))])
    return r

def k1(X, Y, pgen, alpha=1.0, prev='ident', post='ident', **kwargs):
    """Computes the gram matrix.

    :param X: Data matrix where each row is an example and each column a
        categorical attribute.
    :param Y: Data matrix.
    :param pgen: Probability mass function generator (see
        :meth:`~kcat.pgen.get_pgen`).
    :param alpha: Parameter for the inverting function *h*.
    :param prev: Function to transform the data before composing. Accepts
        ``'ident'``, ``'f1'`` or a Python function.
    :param post: Function to transform the data after composing. Accepts
        ``'ident'``, ``'f1'``,  ``'f2'`` or a Python function.
    :param gamma: (optional) Parameter required by ``'f1'`` and  ``'f2'``.

    :returns: Gram matrix obtained applying :meth:`k1_mult` between each pair
        of elements in *X* and *Y*.
    """
    h = lambda x: (1.0 - x ** alpha) ** (1.0 / alpha)
    prevf = get_function(prev, kwargs)
    postf =  get_function(post, kwargs)
    # Compute the kernel matrix:
    G = np.zeros((len(X), len(Y)))
    for i, u in enumerate(X):
        for j, v in enumerate(Y):
            G[i][j] = postf(k1_mult, u, v, h, pgen, prevf)
    return G

def fast_k1(X, Y, pgen, alpha=1.0, prev='ident', post='ident', **kwargs):
    """An optimised version of :meth:`k1` with the same interface.

    Since the code is vectorised any Python functions passed as argument must
    work with numpy arrays.
    """
    h = lambda x: (1.0 - x ** alpha) ** (1.0 / alpha)
    Yp = h(apply_pgen(pgen, Y))
    prevf = get_vector_function(prev, kwargs)
    postf = get_vector_function(post, kwargs) if post != 'f2' else None
    # The function f2 needs to be treated separately.
    xm, xn = X.shape
    ym, yn = Y.shape
    # The gram matrix is computed using vectorised operations because speed:
    XL = np.repeat(X, ym, axis=0)
    YL = np.tile(Y, (xm, 1))
    YP = np.tile(Yp, (xm, 1))
    G = (XL == YL) * YP
    G = np.mean(prevf(G), axis=1)
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
        GX = np.repeat(Px, ym, axis=0)
        GY = np.tile(Yp, (xm, 1))
        GX = np.mean(prevf(GX), axis=1)
        GY = np.mean(prevf(GY), axis=1)
        # Apply f2:
        gamma = kwargs['gamma']
        G = np.exp(gamma * (2.0 * G - GX - GY))
    return G.reshape(xm, ym)


# Categorical Kernel K2

def k2_univ(x, y, p, n):
    """Univariate kernel *K2*.

    :param x: Value.
    :param y: Value.
    :param p: Probability function.
    :param n: Number of elements.
    """
    return 0.0 if x != y else 1.0 / p(x)

def k2_mult(u, v, pgen, n, prev):
    """Multivariate kernel *K2*.

    :param u: Data vector.
    :param v: Data vector.
    :param pgen: Probability mass function generator (see
        :meth:`~kcat.pgen.get_pgen`).
    :param n: Number of elements.
    :param prev: Function to transform the data before composing.

    :returns: Value of applying the kernel :meth:`k2_univ` between each pair of
        attributes in *u* and *v*, and then the composition function.
    """
    # Compute the kernel applying the previous and composition functions:
    r = np.sum([prev(k2_univ(u[i], v[i], pgen(i), n)) for i in range(len(u))])
    return r

def k2(X, Y, pgen, prev='ident', post='ident', **kwargs):
    """Computes the gram matrix.

    :param X: Data matrix where each row is an example and each column a
        categorical attribute.
    :param Y: Data matrix.
    :param pgen: Probability mass function generator (see
        :meth:`~kcat.pgen.get_pgen`).
    :param prev: Function to transform the data before composing. Accepts
        ``'ident'``, ``'f1'`` or a Python function.
    :param post: Function to transform the data after composing. Accepts
        ``'ident'``, ``'f1'``,  ``'f2'`` or a Python function.
    :param gamma: (optional) Parameter required by ``'f1'`` and  ``'f2'``.

    :returns: Gram matrix obtained applying :meth:`k2_mult` between each pair
        of elements in *X* and *Y*.
    """
    prevf = get_function(prev, kwargs)
    postf =  get_function(post, kwargs)
    # Compute the kernel matrix:
    G = np.zeros((len(X), len(Y)))
    for i, u in enumerate(X):
        for j, v in enumerate(Y):
            G[i][j] = postf(np.sqrt(k2_mult(u, v, pgen, len(Y), prevf)))
    return G

def fast_k2(X, Y, pgen, prev='ident', post='ident', **kwargs):
    """An optimised version of :meth:`k2` with the same interface.

    Since the code is vectorised any Python functions passed as argument must
    work with numpy arrays.
    """
    prevf = get_vector_function(prev, kwargs)
    postf = get_vector_function(post, kwargs) if post != 'f2' else None
    # The function f2 needs to be treated separately.
    xm, xn = X.shape
    ym, yn = Y.shape
    Yp = 1.0 / apply_pgen(pgen, Y)
    # The gram matrix is computed using vectorised operations because speed:
    XL = np.repeat(X, ym, axis=0)
    YL = np.tile(Y, (xm, 1))
    YP = np.tile(Yp, (xm, 1))
    G = (XL == YL) * YP
    G = np.sqrt(np.sum(prevf(G), axis=1))
    # When post == 'f2', postf does nothing.
    # The actual post function is applied here:
    if post != 'f2':
        G = postf(G)
    else:
        # We know that: f2 = e ^ (gamma * (2 * k(x, y) - k(x, x) - k(y, y))).
        # The current values of G are those of k(x, y).
        # We need to compute the values of k(x, x) and k(y, y) for each
        # x in X and y in Y:
        Xp = 1.0 / apply_pgen(pgen, X)
        GX = np.repeat(Xp, ym, axis=0)
        GY = np.tile(Yp, (xm, 1))
        GX = np.sqrt(np.sum(prevf(GX), axis=1))
        GY = np.sqrt(np.sum(prevf(GY), axis=1))
        # Apply f2:
        gamma = kwargs['gamma']
        G = np.exp(gamma * (2.0 * G - GX - GY))
    return G.reshape(xm, ym)


# Multivariate

def fast_m1(X, Y, pgen, alpha=1.0, prev='ident', post='ident', **kwargs):
    h = lambda x: (1.0 - x ** alpha) ** (1.0 / alpha)
    prevf = get_vector_function(prev, kwargs)
    postf = get_vector_function(post, kwargs)
    xm, xn = X.shape
    ym, yn = Y.shape
    Xp = h(apply_pgen(pgen, X))
    XL = np.repeat(X, ym, axis=0)
    XP = np.repeat(Xp, ym, axis=0)
    Yp = h(apply_pgen(pgen, Y))
    YL = np.tile(Y, (xm, 1))
    YP = np.tile(Yp, (xm, 1))
    G = prevf((XL == YL) * XP)
    G = np.sum(G, axis=1) * 2 / np.sum(prevf(XP) + prevf(YP), axis=1)
    G = postf(G)
    return G.reshape(xm, ym)


# Expected Likelyhood Kernel

def elk(X, Y):
    xm, xn = X.shape
    ym, yn = Y.shape
    # Compute the kernel matrix:
    X = np.repeat(X, ym, axis=0)
    X *= np.tile(Y, (xm, 1))
    X = np.sqrt(X)
    X = np.sum(X, axis=1)
    return X.reshape(xm, ym)

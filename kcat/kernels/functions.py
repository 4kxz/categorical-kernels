"""This module contains methods to compute the gram matrix using various
kernel functions. The obtained gram matrix can be used to train SVMs
with scikit-learn.

There are two type of methods in this module. The methods
:meth:`k_0_univ`, :meth:`k_0_mult`, :meth:`k_0_matrix`,
:meth:`k_1_univ`, :meth:`k_1_mult`, :meth:`k_1_matrix`,
are meant to be used for testing purposes.
They work with arbitrary functions and collections of iterables, they
can be mixed and swapped to easily test different configurations or
custom transformation function.
However, they are very slow when used on large datasets.

The other methods are all prepared to work on numpy arrays and take
advantage of vectorial operations to speed up computations.
"""

import numpy as np

from ..utils import get_pgen, apply_pgen


# Some of the categorical kernels can receive Python functions as parameters.
# For ease of use, some predefined functions can be specified with a string.
# get_function and get_vector_function take this string and return the
# appropiate function. Both are used when handling the kernel parameters.

def get_function(name, params=None):
    params = {} if params is None else params
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

def get_vector_function(name, params=None):
    params = {} if params is None else params
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


# Non-Vectorised kernels.

def k0_univ(x, y):
    """Univariate kernel *K0* between two values.

    Args:
        x: Category.
        y: Category.

    Returns:
        Value of computing k_1^U(x, y).
    """
    return 0.0 if x != y else 1.0

def k0_mult(u, v, prev):
    """Multivariate kernel *K0* between two vectors.

    Args:
        u: Category vector.
        v: Category vector.
        prev (function): Function to transform the univariate kernel.

    Returns:
        Value of applying the kernel :meth:`k0_univ` between each pair
        of attributes in *u* and *v*, and then the composition function.
    """
    # List comprehension takes care of applying k0 and prev for each element:
    return np.mean([prev(k0_univ, u[i], v[i]) for i in range(len(u))])

def k0_matrix(X, Y, prev='ident', post='ident', **kwargs):
    """Computes the gram matrix between *X* and *Y*.

    Args:
        X: Numpy matrix.
        Y: Numpy matrix.
        prev (string): Function to transform the data before composing.
            Valid values: ``'ident'``, ``'f1'`` or a Python function.
        post (string): Function to transform the data after composing.
            Valid values: ``'ident'``, ``'f1'`` or a Python function.
        gamma (float): Parameter required by ``'f1'``.

    Returns:
        Gram matrix obtained applying :meth:`k0_mult` between each pair
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

def k1_univ(x, y, h, p):
    """Univariate kernel *K1*.

    Args:
        x: Value.
        y: Value.
        h (function): Inverting function.
        p (function): Probability function.

    Returns:
        Value of computing k_1^U(x, y).
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

def k1_matrix(X, Y, pgen, alpha=1.0, prev='ident', post='ident', **kwargs):
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
    postf = get_function(post, kwargs)
    # Compute the kernel matrix:
    G = np.zeros((len(X), len(Y)))
    for i, u in enumerate(X):
        for j, v in enumerate(Y):
            G[i][j] = postf(k1_mult, u, v, h, pgen, prevf)
    return G


# Vectorised kernel functions

def k_0(X, Y, prev='ident', post='ident', **kwargs):
    """Computes a matrix with the values of applying the kernel
    :math:`k_0` between each pair of elements in *X* and *Y*.

    Args:
        X: Numpy matrix.
        Y: Numpy matrix.
        prev (string): Function to transform the data before composing.
            Values: ``'ident'``, ``'f1'`` or a function.
        post (string): Function to transform the data after composing.
            Values: ``'ident'``, ``'f1'`` or a function.
        kwargs (dict): Arguments required by *prev* or *post*.

    Returns:
        Gram matrix with the kernel value between each pair of elements
        in *X* and *Y*.

    Since the code is vectorised any function passed in *prev* or *post*
    must work on numpy arrays.
    """
    prevf = get_vector_function(prev, kwargs)
    postf = get_vector_function(post, kwargs)
    xm, xn = X.shape
    ym, yn = Y.shape
    # The gram matrix is computed using vectorised operations because speed:
    G = np.zeros((xm, ym))
    for i in range(xm):
        Xi = np.tile(X[i], (ym, 1))
        Xi = prevf(Xi == Y)
        G[i, :] = np.mean(Xi, axis=1)
    return postf(G)

def k_1(X, Y, Xp, Yp, alpha=1.0, prev='ident', post='ident', **kwargs):
    """Computes a matrix with the values of applying the kernel
    :math:`k_1` between each pair of elements in *X* and *Y*.

    Args:
        X: Numpy matrix.
        Y: Numpy matrix.
        alpha (float): Argument for the inverting function *h*.
        Xp: Numpy matrix with the probabilities of each category in *X*.
        Yp: Numpy matrix with the probabilities of each category in *Y*.
        prev (string): Function to transform the data before composing.
            Values: ``'ident'``, ``'f1'`` or a function.
        post (string): Function to transform the data after composing.
            Values: ``'ident'``, ``'f1'``,  ``'f2'`` or a function.
        kwargs (dict): Arguments required by *prev* or *post*.

    Returns:
        Gram matrix with the kernel value between each pair of elements
        in *X* and *Y*.

    Since the code is vectorised any function passed in *prev* or *post*
    must work on numpy arrays.
    """
    h = lambda x: (1.0 - x ** alpha) ** (1.0 / alpha)
    prevf = get_vector_function(prev, kwargs)
    postf = get_vector_function(post, kwargs) if post != 'f2' else None
    # The function f2 needs to be treated separately.
    xm, xn = X.shape
    ym, yn = Y.shape
    Yp = h(Yp)
    # The gram matrix is computed using vectorised operations because speed:
    G = np.zeros((xm, ym))
    for i in range(xm):
        Xi = np.tile(X[i], (ym, 1))
        Xi = prevf((Xi == Y) * Yp)
        G[i, :] = np.mean(Xi, axis=1)
    if post != 'f2':
        return postf(G)
    else:
        # We know that: f2 = e ^ (gamma * (2 * k(x, y) - k(x, x) - k(y, y))).
        # The current values of G are those of k(x, y).
        # We need to compute the values of k(x, x) and k(y, y) for each
        # x in X and y in Y:
        gamma = kwargs['gamma']
        Xp = h(Xp)
        GX = np.mean(prevf(Xp), axis=1)
        GX = np.tile(GX, (ym, 1)).T
        GY = np.mean(prevf(Yp), axis=1)
        GY = np.tile(GY, (xm, 1))
        return np.exp(gamma * (2.0 * G - GX - GY))

def k_2(X, Y, Xp, Yp, prev='ident', post='ident', **kwargs):
    """Computes a matrix with the values of applying the kernel
    :math:`k_2` between each pair of elements in *X* and *Y*.

    Args:
        X: Numpy matrix.
        Y: Numpy matrix.
        Xp: Numpy matrix with the probabilities of each category in *X*.
        Yp: Numpy matrix with the probabilities of each category in *Y*.
        prev (string): Function to transform the data before composing.
            Values: ``'ident'``, ``'f1'`` or a function.
        post (string): Function to transform the data after composing.
            Values: ``'ident'``, ``'f1'``,  ``'f2'`` or a function.
        kwargs (dict): Arguments required by *prev* or *post*.

    Returns:
        Gram matrix with the kernel value between each pair of elements
        in *X* and *Y*.

    Since the code is vectorised any function passed in *prev* or *post*
    must work on numpy arrays.
    """
    prevf = get_vector_function(prev, kwargs)
    postf = get_vector_function(post, kwargs) if post != 'f2' else None
    # The function f2 needs to be treated separately.
    xm, xn = X.shape
    ym, yn = Y.shape
    Yp = 1.0 / Yp
    # The gram matrix is computed using vectorised operations because speed:
    G = np.zeros((xm, ym))
    for i in range(xm):
        Xi = np.tile(X[i], (ym, 1))
        Xi = prevf((Xi == Y) * Yp)
        G[i, :] = np.sqrt(np.sum(Xi, axis=1))
    if post != 'f2':
        return postf(G)
    else:
        # We know that: f2 = e ^ (gamma * (2 * k(x, y) - k(x, x) - k(y, y))).
        # The current values of G are those of k(x, y).
        # We need to compute the values of k(x, x) and k(y, y) for each
        # x in X and y in Y:
        gamma = kwargs['gamma']
        Xp = 1.0 / Xp
        GX = np.sqrt(np.sum(prevf(Xp), axis=1))
        GX = np.tile(GX, (ym, 1)).T
        GY = np.sqrt(np.sum(prevf(Yp), axis=1))
        GY = np.tile(GY, (xm, 1))
        return np.exp(gamma * (2.0 * G - GX - GY))

def m_1(X, Y, Xp, Yp, alpha=1.0, prev='ident', post='ident', **kwargs):
    """Computes a matrix with the values of applying the kernel
    :math:`m_1` between each pair of elements in *X* and *Y*.

    Args:
        X: Numpy matrix.
        Y: Numpy matrix.
        Xp: Numpy matrix with the probabilities of each category in *X*.
        Yp: Numpy matrix with the probabilities of each category in *Y*.
        alpha (float): Argument for the inverting function *h*.
        prev (string): Function to transform the data before composing.
            Values: ``'ident'``, ``'f1'`` or a function.
        post (string): Function to transform the data after composing.
            Values: ``'ident'``, ``'f1'``,  ``'f2'`` or a function.
        kwargs (dict): Arguments required by *prev* or *post*.

    Returns:
        Numpy matrix of size :math:`m_x \\times m_y`.

    Since the code is vectorised any function passed in *prev* or *post*
    must work on numpy arrays.
    """
    h = lambda x: (1.0 - x ** alpha) ** (1.0 / alpha)
    prevf = get_vector_function(prev, kwargs)
    postf = get_vector_function(post, kwargs)
    xm, xn = X.shape
    ym, yn = Y.shape
    Xp = h(Xp)
    Yp = h(Yp)
    G = np.zeros((xm, ym))
    for i in range(xm):
        Xi = np.tile(X[i], (ym, 1))
        Xi = prevf((Xi == Y) * Yp)
        Xq = np.tile(Xp[i], (ym, 1))
        Xq = prevf(Xq) + prevf(Yp)
        G[i, :] = 2.0 * np.sum(Xi, axis=1) / np.sum(Xq, axis=1)
    return postf(G)

def elk(X, Y):
    """Computes a matrix with the values of applying the kernel *ELK*
    between each pair of elements in *X* and *Y*.

    Args:
        X: Numpy matrix.
        Y: Numpy matrix.

    Returns:
        Numpy matrix of size :math:`m_x \\times m_y`.
    """
    xm, xn = X.shape
    ym, yn = Y.shape
    # Compute the kernel matrix:
    G = np.zeros((xm, ym))
    for i in range(xm):
        Xi = np.tile(X[i], (ym, 1))
        Xi = np.sqrt(Xi * Y)
        G[i, :] = np.sum(Xi, axis=1)
    return G

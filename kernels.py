import collections

import numpy as np


###############################################################################
# Misc. Functions
###############################################################################

def mean(x):
    """Mean composition function."""
    return sum(x) / len(x)

def prod(x):
    """Product compostion function."""
    p = 1
    for xi in x:
        p *= xi
    return p

ident = lambda k, *args: k(*args)  # Identity

def f1gen(gamma):
    """Returns a "transformation function 1", with the appropiate gamma."""
    return lambda k, x, y, *args: np.exp(gamma *  k(x, y, *args))

def f2gen(gamma):
    """Returns a "transformation function 2", with the appropiate gamma."""
    return lambda k, x, y, *args: np.exp(gamma * (2 * k(x, y, *args) - k(x, x, *args) - k(y, y, *args)))

# Probability mass function:

def get_count(X):
    """
    Returns a `list` of `dict` with the count of class c.
    """
    n = len(X)
    d = len(X[0])
    Y = []
    for i in range(d):
        Y.append(collections.defaultdict(int))  # Initialises entries to 0.
        for j in range(n):
            c = X[j][i]
            Y[i][c] += 1
    return Y

def get_pmf(X, total=False):
    """
    Returns a `list` of `dict` with the probability of class c.
    Probability of the i-th dimension is accessible as pmf[i][c].
    When total is False, calculates the probability for each attribute.
    When total is True, calculates the probability accros the whole dataset.
    """
    n = len(X)
    d = len(X[0])
    t = n * d if total else n
    pmf = []
    for i in range(d):
        pmf.append(collections.defaultdict(int))  # Initialises entries to 0.
        for j in range(n):
            c = X[j][i]
            pmf[i][c] += 1 / t
    return pmf

# pmf is a vector<map<symbol,real>> but to use it would require to pass the
# indices all the way down to the kernel.
# Using a couple of lambdas it can be turned into a function generator that
# returns the appropiate probability function p(c):

def pmf_to_pgen(pmf):
    """
    Returns a generator pgen, such that pgen(i) returns the function p(c) for
    the i-th attribute in `pmf`.
    """
    return lambda i: lambda c: pmf[i][c]


###############################################################################
# Categorical Kernel K0 (Overlap)
###############################################################################

def univ_k0(x, y):
    return 0 if x != y else 1

def mult_k0(u, v, prev, comp):
    """Computes the univariate kernel k0 for each attribute in vectors x and y."""
    # Compute the kernel applying the previous transformation and composition functions:
    return comp([prev(univ_k0, u[i], v[i]) for i in range(len(u))])

def simple_k0(X, Y, prev=ident, comp=mean, post=ident):
    """Returns the gram function for the categorical kernel k0."""
    # Compute the kernel matrix:
    M = np.zeros((len(X), len(Y)))
    for i, xi in enumerate(X):
        for j, yj in enumerate(Y):
            M[i][j] = post(mult_k0, xi, yj, prev, comp)
    return M

# Optimised K0:

def precomp_k0(X, prev=('ident', 1), comp='mean', post=('ident', 1)):
    """
    Returns the gram function for the categorical kernel k0.
    Only works with determined functions.
    """
    d = len(X[0])
    n = len(X)
    # Previous transformation function:
    if prev[0] == 'ident':
        prevf = lambda x: x
    elif prev[0] == 'f1':
        prevf = lambda x: np.exp(prev[1] * x)
    else:
        raise ValueError("Unknown previous transformation function {}.".format(prev))
    # Composition function:
    if comp == 'mean':
        compf = lambda x: np.sum(x, axis=1) / d
    elif comp == 'prod':
        compf = lambda x: np.product(x, axis=1)
    else:
        raise ValueError("Unknown composition function '{}'.".format(comp))
    # Posterior transformation function:
    if post[0] == 'ident':
        postf = lambda x: x
    elif post[0] == 'f1':
        postf = lambda x: np.exp(post[1] * x)
    elif post[0] == 'f2':
        # Since the multivariate kernel uses overlap, k(u, u) == 1
        # independently of whether the composition is the mean or
        # the product, so we simplify:
        postf = lambda x: np.exp(post[1] * (2 * x - 2))
    else:
        raise ValueError("Unknown posterior transformation function {}.".format(post))
    # Compute the kernel matrix:
    M = np.zeros((n, n))
    for i, xi in enumerate(X):
        Xi = np.repeat([xi], n - i, axis=0)
        Mi = X[i:n] == Xi
        Mi = postf(compf(prevf(Mi)))
        M[i, i:n] = M[i:n, i] = Mi
    return M


###############################################################################
# Categorical Kernel K1 (Probabilty Mass Function)
###############################################################################

def univ_k1(x, y, p, h):
    return 0 if x != y else h(p(x))

def mult_k1(u, v, pgen, h, prev, comp):
    """Computes the univariate kernel k1 for each attribute in vectors x and y."""
    # Compute the kernel applying the previous transformation and composition functions:
    return comp([prev(univ_k1, u[i], v[i], pgen(i), h) for i in range(len(u))])

def simple_k1(X, Y, pmf=None, alpha=1, prev=ident, comp=mean, post=ident):
    """Returns the gram function for the categorical kernel k1."""
    # When pmf is unknown compute it from X:
    if pmf is None:
        pmf = get_pmf(X)
    pgen = pmf_to_pgen(pmf)
    # Inverting function h_a:
    h = lambda x: (1 - x ** alpha) ** (1 / alpha)
    # Compute the kernel matrix:
    M = np.zeros((len(X), len(Y)))
    for i, xi in enumerate(X):
        for j, yj in enumerate(Y):
            M[i][j] = post(mult_k1, xi, yj, pgen, h, prev, comp)
    return M

# Optimised K1:

def precomp_k1(X, pmf=None, alpha=1, prev=('ident', None), comp='mean', post=('ident', None)):
    """
    Returns the gram function for the categorical kernel k1.
    Only works with determined functions.
    """
    d = len(X[0])
    n = len(X)
    if pmf is None:
        pmf = get_pmf(X)
    # Inverting function h:
    h = lambda x: (1 - x ** alpha) ** (1 / alpha)
    # Previous transformation function:
    if prev[0] == 'ident':
        prevf = lambda x: x
    elif prev[0] == 'f1':
        prevf = lambda x: np.exp(prev[1] * x)
    else:
        raise ValueError("Unknown previous transformation function {}.".format(prev))
    # Composition function:
    if comp == 'mean':
        compf = lambda x: np.sum(x, axis=1) / d
    elif comp == 'prod':
        compf = lambda x: np.product(x, axis=1)
    else:
        raise ValueError("Unknown composition function '{}'.".format(comp))
    # Posterior transformation function:
    if post[0] == 'ident':
        postf = lambda x: x
    elif post[0] == 'f1':
        postf = lambda x: np.exp(post[1] * x)
    elif post[0] == 'f2':
        postf = lambda x: x  # The post f2 is applied later in a seperate loop...
    else:
        raise ValueError("Unknown posterior transformation function {}.".format(post))
    # Create a matrix with the weights, for convenience:
    P = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            v = X[i][j]
            P[i][j] = h(pmf[j][v])  # Pre-apply h(x) to all the categories in pmf.
    # Compute the kernel matrix:
    M = np.zeros((n, n))
    for i in range(n):
        Xi = np.repeat([X[i]], n - i, axis=0)
        Pi = np.repeat([P[i]], n - i, axis=0)
        Mi = (X[i:n] == Xi) * Pi
        Mi = postf(compf(prevf(Mi)))
        M[i, i:n] = M[i:n, i] = Mi
    # When post == 'f2', postf does nothing. The actual post function is applied here:
    if post[0] == 'f2':
        # We know that: f2 = e ^ (gamma * (2 * k(x, y) - k(x, x) - k(y, y))).
        # The current values of M are those of k(x, y).
        # The values of k(z, z) are computed in the diagonal of M for any z.
        # We can use that to avoid recomputing k(x, x) and k(y, y):
        kxx = np.diag(M)
        kyy = kxx[np.newaxis].T
        M = np.exp(post[1] * (2 * M - kxx - kyy))
    return M


###############################################################################
# Categorical Kernel K2 (Chi-Square)
###############################################################################

def precomp_k2(X, pmf=None):
    d = len(X[0])
    n = len(X)
    if pmf is None:
        pmf = get_pmf(X)
    # Create a matrix with the weights, for convenience:
    P = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            v = X[i][j]
            P[i][j] = 1 / pmf[j][v]
    # Compute the kernel matrix:
    M = np.zeros((n, n))
    for i in range(n):
        Xi = np.repeat([X[i]], n - i, axis=0)
        Pi = np.repeat([P[i]], n - i, axis=0)
        Mi = (X[i:n] == Xi) * Pi
        Mi = Mi.sum(axis=1) / d
        M[i, i:n] = M[i:n, i] = Mi
    return M


###############################################################################
# WIP
###############################################################################

# def precomp_true_chi(X, pmf=None):
#     d = len(X[0])
#     n = len(X)
#     if pmf is None:
#         pmf = get_pmf(X, total=True)
#     P = np.zeros((n, d))
#     for i in range(n):
#         for j in range(d):
#             v = X[i][j]
#             P[i][j] = pmf[j][v]
#     c = P.sum(axis=0)
#     r = P.sum(axis=1)
#     E = c * r[np.newaxis].T
#     M = np.zeros((n, n))
#     for i in range(n):
#         Xi = np.repeat([X[i]], n - i, axis=0)
#         Pi = np.repeat([P[i]], n - i, axis=0)
#         Ei = np.repeat([E[i]], n - i, axis=0)
#         Mi = (P[i:n] - E[i:n]) ** 2 / E[i:n] + (Pi - Ei) ** 2 / Ei
#         Mi = Mi.mean(axis=1)
#         Mi = np.exp(-Mi)
#         M[i, i:n] = M[i:n, i] = Mi
#     return M

# def precomp_elk(X, pmf=None):
#     d = len(X[0])
#     n = len(X)
#     if pmf is None:
#         pmf = get_pmf(np.transpose(X))
#     # Create a matrix with the inverted weigts, for convenience:
#     C = np.zeros((n, d))
#     for i in range(n):
#         for j in range(d):
#             v = X[i][j]
#             C[i][j] = 1 / pmf[i][v]
#     # Compute the kernel matrix:
#     M = np.zeros((n, n))
#     for i, xi in enumerate(X):
#         Mi = np.repeat([X[i]], n - i, axis=0)
#         Ci = np.repeat([C[i]], n - i, axis=0)
#         Ei = abs(Mi - X[i:]) == 0
#         Ei = Ei * Ci * C[i:]
#         Ei = np.sum(Ei, axis=1)
#         M[i, i:] = M[i:, i] = Ei
#     return M


###############################################################################
# TEST KERNELS
###############################################################################

if __name__ == '__main__':
    x = np.array([['a', 'A'], ['b', 'B'], ['c', 'A']])
    a = simple_k0(x, x)
    b = precomp_k0(x)
    print((a == b).all())
    a = simple_k0(x, x, post=f1gen(2))
    b = precomp_k0(x, post=('f1', 2))
    print((a == b).all())
    a = simple_k0(x, x, post=f2gen(2))
    b = precomp_k0(x, post=('f2', 2))
    print((a == b).all())
    a = simple_k1(x, x)
    b = precomp_k1(x)
    print((a == b).all())
    a = simple_k1(x, x, post=f1gen(2))
    b = precomp_k1(x, post=('f1', 2))
    print((a == b).all())
    print(precomp_k0(x))
    print(precomp_k1(x))
    print(precomp_k2(x))

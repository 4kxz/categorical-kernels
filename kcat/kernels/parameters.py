"""This module defines a number of constants. They're parameter dictionaries
with default values that can be directly passed to GridSearch, which in turn
can be used to find the best values.

The search space defined in the dictioanries should work reasonably well
in most cases and avoids having to define a search dictionary every single
time.
"""

import numpy as np

np.set_printoptions(precision=2, threshold=4, edgeitems=2)


PARAMS_RBF = {
    'param_grid': {
        'C': 10.0 ** np.arange(-1, 3),
        'gamma': 2.0 ** np.arange(-12, 1),
    },
}  #: Default parameters for RBF kernel.


PARAMS_K0 = {
    'param_grid': {
        'C': 10.0 ** np.arange(-1, 3),
    },
    'functions': [
        ('ident', 'ident'),
        ('ident', 'f1'),
        ('f1', 'ident'),
    ],
    'gammas': 2.0 ** np.arange(-3, 3),
}  #: Default parameters for K0 kernel.


PARAMS_K1 = {
    'param_grid': {
        'C': 10.0 ** np.arange(-1, 3),
    },
    'alphas': 1.5 ** np.arange(-4, 3),
    'functions': [
        ('ident', 'ident'),
        ('ident', 'f1'),
        ('ident', 'f2'),
        ('f1', 'ident'),
    ],
    'gammas': 2.0 ** np.arange(-3, 3),
}  #: Default parameters for K1 kernel.


PARAMS_K2 = {
    'param_grid': {
        'C': 10.0 ** np.arange(-1, 3),
    },
    'functions': [
        ('ident', 'ident'),
        ('ident', 'f1'),
        ('ident', 'f2'),
        ('f1', 'ident'),
    ],
    'gammas': 2.0 ** np.arange(-3, 1),
}  #: Default parameters for K2 kernel.


PARAMS_ELK = {
    'param_grid': {
        'C': 10.0 ** np.arange(-1, 3),
    },
}  #: Default parameters for ELK kernel.

PARAMS_M0 = {
    'param_grid': {
        'C': 10.0 ** np.arange(-1, 3),
    },
    'alphas': 1.5 ** np.arange(-4, 3),
}  #: Default parameters for M0 kernel.

PARAMS_M1 = {
    'param_grid': {
        'C': 10.0 ** np.arange(-1, 3),
    },
}  #: Default parameters for M1 kernel.

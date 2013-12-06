import numpy as np


PARAMS_RBF = {
    'param_grid': {
        'C': 10.0 ** np.arange(-2, 8),
        'gamma': 2.0 ** np.arange(-15, 1),
    },
}


PARAMS_K0 = {
    'param_grid': {
        'C': 10.0 ** np.arange(-2, 8),
    },
    'functions': [
        ('ident', 'ident'),
        ('ident', 'f1'),
        ('f1', 'ident'),
    ],
    'gammas': 2.0 ** np.arange(-6, 2),
}


PARAMS_K1 = {
    'param_grid': {
        'C': 10.0 ** np.arange(-2, 8),
    },
    'alphas': 2.0 ** np.arange(-4, 2),
    'functions': [
        ('ident', 'ident'),
        ('ident', 'f1'),
        ('ident', 'f2'),
        ('f1', 'ident'),
    ],
    'gammas': 2.0 ** np.arange(-6, 2),
}


PARAMS_K2 = {
    'param_grid': {
        'C': 10.0 ** np.arange(-2, 8),
    },
}

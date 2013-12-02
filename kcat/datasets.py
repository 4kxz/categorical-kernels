import urllib.request
import collections

import numpy as np
from numpy import random


def synthetic(m, n=25, c=4, p=0.5, encoder=False, random_state=None):
    """
    :param m: Number of examples to generate.
    :param n: Number of attributes for each example.
    :param c: Number of classes.
    :param p: Change probability of class-unique attributes.
    :type p: float

    The effect of the parameter `p` according to its value:

    - *p* close to 0: Class attributes almost never happen (random/harder).
    - *p* close to 1: Class attributes happen as often as the rest (easier).

    :returns: - A matrix of size :math:`m \\times n` with the dataset.
              - An array with the class of the `m` examples.
              - (optional) A binary encoding function for X.
    """
    random.seed(random_state)
    p **= 2.0  # Makes the effect of the parameter more linear.
    a = c + 2  # Number of attribute values.
    # Assign class to each example:
        # y = random.randint(c, size=m)
    k = [float(i % c) for i in range(m)]
    y = random.choice(k, m, replace=False)
    # Generate attributes:
    X = np.zeros((m, n))
    for j in range(n):
        # Generate a list of values:
        values = np.arange(a)
        # Pick one value at random for each class:
        unique = random.choice(values, c, replace=False)
        # The rest are common attributes:
        common = list(set(values) - set(unique))
        # Generate attributes:
        for i in range(m):
            # Choose to assing a random attribute or a unique one:
            rand = random.random() > (c / a * p)
            value = random.choice(common) if rand else unique[y[i]]
            X[i][j] = value
    # When encoder is True, return a binary encoder for the dataset. The
    # encoder is necessary because small datasets may not contain instances of
    # every single category in the dataset, making it impossible to infer the
    # correct binarization from just part of the whole dataset.
    if not encoder:
        return X, y
    else:
        def binary_encoder(X):
            # Each attribute has a value between 0 and `a`.
            # We create the matrix with zeros, with a column for each possible
            # category of each attribute, then set to one the corresponding
            # column for each example.
            Y = np.zeros((m, n * a))
            # For each attribute:
            for i in range(m):
                for j in range(n):
                    v = X[i][j]
                    Y[i][j * a + v] = 1
            return Y
        return X, y, binary_encoder


def gmonks(m, d=3, random_state=None):
    """
    :param m: Number of examples to generate.
    :param d: Number of blocks of features for each example.

    Each block is a set of six features generated according to the description
    in the original monks problem.

    :returns: - A matrix of size :math:`m \\times (6d)` with the dataset.
              - An array with the class of the `m` examples.
    """
    random.seed(random_state)
    # Give names to the categories, makes the code easier to read:
    C1, C2, C3, C4 = range(4)
    categories = (
        (C1, C2, C3),
        (C1, C2, C3),
        (C1, C2),
        (C1, C2, C3),
        (C1, C2, C3, C4),
        (C1, C2),
    )
    x, y = [], []
    for i in range(m):
        x.append([])
        fk = 0
        for j in range(d):
            fj = [random.choice(a) for a in categories]
            p1 = fj[0] == fj[1] or fj[4] == C1
            p2 = sum(1 if x == C1 else 0 for x in fj) >= 2
            p3 = (fj[4] == C3 and fj[3] == C1) or (fj[4] != C3 and fj[1] != C2)
            fk += 1 if p2 and not(p1 and p3) else 0
            x[i] += fj
        y.append(fk >= d / 2)
    return np.array(x), np.array(y)


def promoters(random_state=None):
    """
    Downloads the promoter gene sequences dataset from the internet and loads
    them into a data set.

    :returns: - A matrix of size :math:`106 \\times 57` with the dataset.
              - An array with the class of the `106` examples.
    """
    # TODO: Add shuffle.
    random.seed(random_state)
    with urllib.request.urlopen(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/'
        'molecular-biology/promoter-gene-sequences/promoters.data'
    ) as promoters_data:
        categories = {'a': 0, 'c': 1, 'g': 2, 't': 3}
        X, y = [], []
        for line in promoters_data:
            cat, _, seq = line.decode('ascii').split(',')
            X.append([categories[i] for i in seq.strip()])
            y.append(cat == '+')
        X, y = np.array(X), np.array(y)
        return X,  y


def soybean(random_state=None):
    """
    Downloads the soybean dataset from the internet.

    :returns: - A matrix of size :math:`? \\times ?` with the dataset.
              - An array with the class of the `?` examples.
    """
    # TODO: Add shuffle.
    random.seed(random_state)
    with urllib.request.urlopen(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/'
        'soybean/soybean-large.data'
    ) as promoters_data:
        X, y = [], []
        encode = lambda x: 0 if x == '?' else int(x) + 1
        for line in promoters_data:
            seq = line.decode('ascii').split(',')
            y.append(seq[0])
            X.append([encode(x.strip()) for x in seq[2:]])
        X, y = np.array(X), np.array(y)
        return X,  y


def soybean_test(random_state=None):
    # TODO: Add shuffle.
    random.seed(random_state)
    with urllib.request.urlopen(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/'
        'soybean/soybean-large.test'
    ) as promoters_data:
        X, y = [], []
        encode = lambda x: 0 if x == '?' else int(x) + 1
        for line in promoters_data:
            seq = line.decode('ascii').split(',')
            y.append(seq[0])
            X.append([encode(x.strip()) for x in seq[2:]])
        X, y = np.array(X), np.array(y)
        return X,  y

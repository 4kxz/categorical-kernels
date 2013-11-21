import random
import urllib.request

import numpy as np


def synthetic(n, d=20, c=2, p=0.5):
    """
    * `n`: number of examples to generate.
    * `d`: number of attributes for each example.
    * `c`: number of classes.
    * `p`: change probability of class-unique attributes:
        * `p` close to 0: class attributes almost never happen (random/harder).
        * `p` close to 1: class attributes happen as often as the rest (easier).
    """
    p **= 2  # Makes the effect of the parameter more linear.
    a = c + 2  # Number of attribute values.
    # Assign class to each example:
    y = [i % c for i in range(n)]
    # Generate attributes:
    x = [[] for _ in range(n)]
    for i in range(d):
        # Generate a list of values:
        common = list(range(a))
        # Pick one value at random for each class:
        unique = [common.pop(random.randrange(len(common))) for _ in range(c)]
        # Generate attributes:
        for j in range(n):
            # Choose to assing a random attribute or a unique one:
            rand = random.random() > c / a * p
            value = random.choice(common) if rand else unique[y[j]]
            x[j].append(value)
    return np.array(x), np.array(y)


def gmonks(n, d):
    """
    Returns a matrix with `n` examples of the monks dataset as rows, which have
    `d` * 6 features each. Also returns a vector of `n` booleans with the class
    of each example.
    """
    # Give names to the categories, makes the code easier to read:
    C1, C2, C3, C4, C5, C6 = range(6)
    categories = (
        (C1, C2, C3),
        (C1, C2, C3),
        (C1, C2),
        (C1, C2, C3),
        (C1, C2, C3, C4),
        (C1, C2, C3, C4, C5, C6),
    )
    x, y = [], []
    for i in range(n):
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


def promoters():
 with urllib.request.urlopen(
    'http://archive.ics.uci.edu/ml/machine-learning-databases/'
    'molecular-biology/promoter-gene-sequences/promoters.data'
) as promoters_data:
    categories = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    x, y = [], []
    for line in promoters_data:
        cat, _, seq = line.decode('ascii').split(',')
        x.append([categories[i] for i in seq.strip()])
        y.append(cat == '+')
    return np.array(x), np.array(y)

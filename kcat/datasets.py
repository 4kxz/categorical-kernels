import urllib.request

import numpy as np
from numpy import random
from sklearn import cross_validation as cv

from .utils import dummy_variable


class Dataset:

    def __init__(self, *args, **kwargs):
        self.data_arrays = self.generate(*args, **kwargs)

    def generate(self):
        raise NotImplementedError

    def train_test_split(self, **kwargs):
        split = cv.train_test_split(*self.data_arrays, **kwargs)
        C_tr, C_ts, Q_tr, Q_ts, y_train, y_test = split
        X_train = {'categorical': C_tr, 'default': Q_tr}
        X_test = {'categorical': C_ts, 'default': Q_ts}
        return X_train, X_test, y_train, y_test


class Synthetic(Dataset):

    def generate(self, m, n=25, c=4, p=0.5, random_state=None):
        """Generates a random data set.

        :param m: Number of examples to generate.
        :type m: int
        :param n: Number of attributes for each example.
        :type n: int
        :param c: Number of classes.
        :type c: int
        :param p: Change probability of class-unique attributes.
        :type p: float

        The effect of the *p* according to its value:

        - *p* close to 0: Class attributes almost never happen (random/harder).
        - *p* close to 1: Class attributes happen as often as the rest (easier).

        :returns: - A matrix of size :math:`m \\times n` with the dataset.
                  - An array with the class of the `m` examples.
        """
        random.seed(random_state)
        p **= 2.0  # Makes the effect of the parameter more linear.
        a = c + 2  # Number of attribute values.
        # Assign class to each example:
        y = random.randint(c, size=m)
        # k = [float(i % c) for i in range(m)]
        # y = random.choice(k, m, replace=False)
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
        # Encode the dataset in dummy variable form:
        # Each attribute has a value between 0 and `a`.
        # We create the matrix with zeros, with a column for each possible
        # category of each attribute, then set to one the corresponding column
        # for each example.
        Y = np.zeros((m, n * a))
        # For each attribute:
        for i in range(m):
            for j in range(n):
                v = X[i][j]
                Y[i][j * a + v] = 1
        return X, Y, y


class GMonks(Dataset):

    def generate(self, m, d=2, random_state=None):
        """Generates a random data set.

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
        X, y = [], []
        for i in range(m):
            X.append([])
            fk = 0
            for j in range(d):
                fj = [random.choice(a) for a in categories]
                p1 = fj[0] == fj[1] or fj[4] == C1
                p2 = sum(1 if x == C1 else 0 for x in fj) >= 2
                p3 = (fj[4] == C3 and fj[3] == C1) or (fj[4] != C3 and fj[1] != C2)
                fk += 1 if p2 and not(p1 and p3) else 0
                X[i] += fj
            y.append(fk >= d / 2)
        X, y = np.array(X), np.array(y)
        return X, dummy_variable(X), y


class promoter(Dataset):

    def generate(self):
        """Downloads the promoter gene sequences dataset and loads them into a data
        set. `Source <http://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28\
        Promoter+Gene+Sequences%29>`_.

        :returns: - A matrix of size :math:`106 \\times 57` with the dataset.
                  - An array with the class of the 106 examples.
        """
        with urllib.request.urlopen(
            'http://archive.ics.uci.edu/ml/machine-learning-databases/'
            'molecular-biology/promoter-gene-sequences/promoters.data'
        ) as data:
            categories = {'a': 0, 'c': 1, 'g': 2, 't': 3}
            X, y = [], []
            for line in data:
                cat, _, seq = line.decode('ascii').split(',')
                X.append([categories[i] for i in seq.strip()])
                y.append(cat == '+')
            X, y = np.array(X), np.array(y)
            return X, dummy_variable(X), y


class Splice(Dataset):

    def generate(self):
        """Downloads the splice junction gene sequences dataset and loads them into
        a data set. `Source <http://archive.ics.uci.edu/ml/datasets/Molecular+Biol\
        ogy+%28Splice-junction+Gene+Sequences%29>`_.

        :returns: - A matrix of size :math:`3190 \\times 60` with the dataset.
                  - An array with the class of the 3190 examples.
        """
        with urllib.request.urlopen(
            'http://archive.ics.uci.edu/ml/machine-learning-databases/'
            'molecular-biology/splice-junction-gene-sequences/splice.data'
        ) as data:
            categories = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'D': 4, 'N': 5, 'S': 6, 'R': 7}
            classes = {'EI': 0, 'IE': 1, 'N': 2}
            X, y = [], []
            for line in data:
                cat, _, seq = line.decode('ascii').split(',')
                X.append([categories[i] for i in seq.strip()])
                y.append(classes[cat])
            X, y = np.array(X), np.array(y)
            return X, dummy_variable(X), y


class Soybean(Dataset):

    def generate(self):
        """Downloads the soybean dataset. Includes train and test.
        The last four classes are ignored (see source for explanation). `Source <h\
        ttp://archive.ics.uci.edu/ml/datasets/Soybean+%28Large%29>`_.

        :returns: - A matrix of size :math:`630 \\times ?` with the dataset.
                  - An array with the class of the 630 examples.
        """
        X, y = [], []
        encode = lambda x: 0 if x == '?' else int(x) + 1
        train_size = 0
        with urllib.request.urlopen(
            'http://archive.ics.uci.edu/ml/machine-learning-databases/'
            'soybean/soybean-large.data'
        ) as data:
            for line in data:
                seq = line.decode('ascii').split(',')
                y.append(seq[0])
                X.append([encode(x.strip()) for x in seq[1:]])
                train_size += 1
                if train_size == 290:
                    break
        test_size = 0
        with urllib.request.urlopen(
            'http://archive.ics.uci.edu/ml/machine-learning-databases/'
            'soybean/soybean-large.test'
        ) as data:
            for line in data:
                seq = line.decode('ascii').split(',')
                y.append(seq[0])
                X.append([encode(x.strip()) for x in seq[1:]])
                test_size += 1
                if test_size == 340:
                    break
        X, y = np.array(X), np.array(y)
        return X, dummy_variable(X), y


class Mushroom(Dataset):

    def generate(self):
        """Downloads the mushroom dataset.
        `Source <http://archive.ics.uci.edu/ml/datasets/Mushroom>`_.

        :returns: - A matrix of size :math:`8124 \\times 22` with the dataset.
                  - An array with the class of the 8124 examples.
        """
        X, y = [], []
        with urllib.request.urlopen(
            'http://archive.ics.uci.edu/ml/machine-learning-databases/'
            'mushroom/agaricus-lepiota.data'
        ) as data:
            for line in data:
                seq = line.decode('ascii').split(',')
                y.append(seq[0])
                X.append([ord(x.strip()) for x in seq[1:]])
        X, y = np.array(X), np.array(y)
        return X, dummy_variable(X), y


class WebKB(Dataset):

    def generate(self):
        """Downloads the webkb dataset.
        `Source <http://web.ist.utl.pt/~acardoso/datasets/>`_.
        """
        X, y, ref, index = [], [], [], {}
        with urllib.request.urlopen(
            'http://localhost:8000/kcat/data/webkb-stemmed.txt'
        ) as data:
            for line in data:
                seq = line.decode('ascii').replace('\n', '').split(' ')
                label, row = seq[0], seq[1:]
                if label in ('student', 'faculty') and len(row) != 0:
                    y.append(label)
                    X.append(row)
                    for word in row:
                        if word not in index:
                            index[word] = len(ref)
                            ref.append(word)
        # Create empty count matrix with as many columns as words and fill it
        C = np.zeros((len(X), len(ref)))
        for i in range(len(X)):
            for word in X[i]:
                C[i][index[word]] += 1
        # Transform frequency to inverse by row
        C /= C.sum(axis=1, keepdims=True)
        X, y = np.array(C), np.array(y)
        categorize = lambda x: 0 if x < 0.0001 else \
                               1 if x < 0.001 else \
                               2 if x < 0.01 else \
                               3 if x < 0.1 else 4
        categorize = np.vectorize(categorize)
        return categorize(X), X, y

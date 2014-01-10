#!/usr/bin/env python

from argparse import ArgumentParser
import json
import time

from sklearn import cross_validation as cv

from kcat.datasets import synthetic
from kcat.kernels import train_test as tt
from kcat.kernels.utils import get_pgen


rbf = tt.TrainTestRBF()
k0 = tt.TrainTestK0()
k1 = tt.TrainTestK1()
k2 = tt.TrainTestK2()


class ExperimentArgumentParser(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            '-m', '--size',
            default='300',
            )
        self.add_argument(
            '-i', '--iterations',
            default='1',
            )


class SyntheticArgumentParser(ExperimentArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            '-n', '--attributes',
            default='25',
            )
        self.add_argument(
            '-c', '--classes',
            default='4',
            )
        self.add_argument(
            '-p', '--p-range',
            action='store_true',
            )


def run(dataset, data_args, split=2/3, folds=5, random_state=None):
    X, y, bincoder = dataset(random_state=random_state, **data_args)
    # Split the data in train and test:
    X_train, X_test, y_train, y_test = \
        cv.train_test_split(X, y, test_size=split, random_state=random_state)
    # Compute other stuff needed for some kernels:
    Xb_train = bincoder(X_train)
    Xb_test = bincoder(X_test)
    pgen = get_pgen(X_train)
    # Specify the cross-validation to use:
    cvf = cv.StratifiedKFold(y_train, folds)
    # Test preformance:
    kernels = {}
    print('Training rbf...')
    kernels['rbf'] = rbf.train_test(cvf, Xb_train, y_train, Xb_test, y_test)
    print('Training k0...')
    kernels['k0'] = k0.train_test(cvf, X_train, y_train, X_test, y_test)
    print('Training k1...')
    kernels['k1'] = k1.train_test(cvf, X_train, y_train, X_test, y_test, pgen=pgen)
    print('Training k2...')
    kernels['k2'] = k2.train_test(cvf, X_train, y_train, X_test, y_test, pgen=pgen)
    return {
        'timestamp': time.asctime(),
        'data_args': data_args,
        'kernels': kernels,
        }


def run_experiment(args):
    m = int(args.size)
    n = int(args.attributes)
    c = int(args.classes)
    results = []
    for i in range(int(args.iterations)):
        print(time.asctime())
        if args.p_range:
            for p in np.arange(0.2, 0.8, 0.1):
                data_args = dict(m=m, n=n, c=c, p=p)
                results.append(run(random_state=i, dataset=synthetic, data_args=data_args))
        else:
            data_args = dict(m=m, n=n, c=c, p=0.5)
            results.append(run(random_state=i, dataset=synthetic, data_args=data_args))
    return results

args = SyntheticArgumentParser().parse_args()

results = run_experiment(args)

with open("comparison-results-2.json", "w+") as f:
    f.write(json.dumps(results))

import json
import time

import numpy as np
from sklearn import cross_validation as cv

from kcat import datasets
from kcat.kernels import models
from kcat.utils import get_pgen


class BaseRunner:
    """Class that abstracts the process of generating a dataset
    and doing training and test with it.

    *BaseRunner* takes care of common code. Code particular to one
    dataset can be implemented by subclassing *BaseRunner*.
    """

    def __init__(self, state):
        self.state = state
        self.results = []

    def _generate_dataset(self, args):
        """Returns a new dataset. To be overwritten by subclasses."""
        raise NotImplementedError

    def _single_run(self, args):
        """Generate a dataset an train/test all the kernels on it."""
        if args.verbose:
            print("#{} {}".format(self.state, time.asctime()))
        # Use the appropiate dataset:
        dataset, data_args = self._generate_dataset(args=args)
        X, y, bincoder = dataset
        # Split the data in train and test:
        X_train, X_test, y_train, y_test = cv.train_test_split(X, y,
            train_size=args.train_size,
            test_size=args.test_size,
            random_state=self.state,
            )
        # Compute other stuff needed for some kernels:
        Xb_train = bincoder(X_train)
        Xb_test = bincoder(X_test)
        pgen = get_pgen(X_train)
        # Specify the cross-validation to use:
        cvf = cv.StratifiedKFold(y_train, args.folds)
        # Test preformance with every kernel:
        kernels = {}
        if args.verbose:
            print('Running rbf...')
        kernels['rbf'] = models.RBF.evaluate(cvf, Xb_train, Xb_test, y_train, y_test)
        if args.verbose:
            print('Running k0...')
        kernels['k0'] = models.K0.evaluate(cvf, X_train, X_test, y_train, y_test)
        if args.verbose:
            print('Running k1...')
        kernels['k1'] = models.K1.evaluate(cvf, X_train, X_test, y_train, y_test, pgen=pgen)
        if args.verbose:
            print('Running k2...')
        kernels['k2'] = models.K2.evaluate(cvf, X_train, X_test, y_train, y_test, pgen=pgen)
        if args.verbose:
            print('Running m1...')
        kernels['m1'] = models.M1.evaluate(cvf, X_train, X_test, y_train, y_test, pgen=pgen)
        if args.verbose:
            print('Running elk...')
        kernels['elk'] = models.ELK.evaluate(cvf, X_train, X_test, y_train, y_test)
        # Update stuff and return results:
        self.state += 1
        self.results.append({
            'timestamp': time.asctime(),
            'run_args': dict(args._get_kwargs()),
            'data_args': data_args,
            'kernels': kernels,
            })
        # Save partial results when specified:
        if args.tmp:
            self.save("{}~".format(args.output))

    def _batch_run(self, args):
        """Generate datasets and train/test repeatedly."""
        for i in range(args.iterations):
            self._single_run(args)

    def run(self, args):
        self._batch_run(args)

    def save(self, filename):
        with open(filename, "w+") as f:
            f.write(json.dumps(self.results))


class SyntheticRunner(BaseRunner):
    """Batch run train/test with the synthetic dataset."""

    def _generate_dataset(self, args):
        data_args = {
            'm': args.test_size + args.train_size,
            'n': args.n,
            'c': args.c,
            'p': args.p,
            'random_state': self.state,
        }
        return datasets.synthetic(**data_args), data_args

    def run(self, args):
        if args.p_range:
            for p in np.arange(0.1, 1.0, 0.1):
                args.p = p;
                super()._batch_run(args)
        else:
            super()._batch_run(args)


class GmonksRunner(BaseRunner):
    """Batch run train/test with the gmonks dataset."""

    def _generate_dataset(self, args):
        data_args = {
            'm': args.test_size + args.train_size,
            'd': args.d,
            'random_state': self.state,
        }
        return datasets.gmonks(**data_args), data_args

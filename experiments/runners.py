import json
import time

import numpy as np
from sklearn import cross_validation as cv

from kcat import datasets as ds
from kcat.kernels import models as md
from kcat.utils import get_pgen


class BaseRunner:
    """Class that abstracts the process of generating a dataset
    and doing training and test with it.

    *BaseRunner* takes care of common code. Code particular to one
    dataset can be implemented by subclassing *BaseRunner*.
    """

    def __init__(self, state, verbose):
        self.state = state
        self.verbose = verbose
        self.results = []

    def save(self, filename):
        with open(filename, "w+") as f:
            f.write(json.dumps(self.results))

    def run(self, **kwargs):
        self._batch_run(**kwargs)

    def _batch_run(self, iterations, **kwargs):
        """Generate datasets and train/test repeatedly."""
        for i in range(iterations):
            self._single_run(**kwargs)

    def _single_run(self, **kwargs):
        """Generate a dataset an train/test all the kernels on it."""
        if self.verbose:
            print("#{} {}".format(self.state, time.asctime()))
        # Generate the appropiate dataset:
        dataset = self._generate_dataset(**kwargs)
        # Split the data in train and test:
        X_train, X_test, y_train, y_test = dataset.train_test_split(
            train_size=kwargs['train_size'],
            test_size=kwargs['test_size'],
            random_state=self.state,
            )
        # Cross validation folds:
        cvf = cv.StratifiedKFold(y_train, kwargs['folds'])
        # Test preformance with every kernel:
        evaluation = {}
        for kernel_model in md.train_test_models:
            # Train the model:
            if self.verbose:
                print('Training {}...'.format(kernel_model.__name__))
            model = kernel_model.train(cv=cvf, X=X_train, y=y_train)
            # Test the model:
            if self.verbose:
                print('Testing {}...'.format(kernel_model.__name__))
            results = kernel_model.test(model, X=X_test, y=y_test)
            evaluation[kernel_model.__name__] = results
        # Update results:
        self.results.append({
            'timestamp': time.asctime(),
            'arguments': kwargs,
            'evaluation': evaluation,
            })
        # Save partial results:
        if self.verbose:
            self.save("{}.json~".format(kwargs['dataset']))
        # Change state for next run:
        self.state += 1

    def _generate_dataset(self, **kwargs):
        """Returns a new dataset. To be overwritten by subclasses."""
        raise NotImplementedError


class SyntheticRunner(BaseRunner):
    """Batch run train/test with the synthetic dataset."""

    def run(self, p_range, **kwargs):
        if p_range:
            for p in np.arange(0.1, 1.0, 0.1):
                kwargs['p'] = p;
                super()._batch_run(**kwargs)
        else:
            super()._batch_run(**kwargs)

    def _generate_dataset(self, train_size, test_size, n, c, p, **kwargs):
        m = train_size + test_size
        return ds.Synthetic(m=m, n=n, c=c, p=p, random_state=self.state)


class GMonksRunner(BaseRunner):
    """Batch run train/test with the GMonks dataset."""

    def _generate_dataset(self, train_size, test_size, d, **kwargs):
        m = train_size + test_size
        return ds.GMonks(m=m, d=d, random_state=self.state)


class WebKBRunner(BaseRunner):
    """Batch run train/test with the WebKB dataset."""

    def _generate_dataset(self, **kwargs):
        return ds.WebKB()

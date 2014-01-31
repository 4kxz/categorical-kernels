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

    def __init__(self, state, tmp=False, verbose=False):
        self.state = state
        self.tmp = tmp
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

    def _single_run(self, train_size, test_size, folds, **kwargs):
        """Generate a dataset an train/test all the kernels on it."""
        if self.verbose:
            print("#{} {}".format(self.state, time.asctime()))
        # Generate the appropiate dataset:
        size = train_size + test_size
        dataset = self._generate_dataset(size=size, **kwargs)
        # Split the data in train and test:
        X_train, X_test, y_train, y_test = dataset.train_test_split(
            train_size=train_size,
            test_size=test_size,
            random_state=self.state,
            )
        # Cross validation folds:
        cvf = cv.StratifiedKFold(y_train, folds)
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
        # Save partial results when specified:
        if self.tmp:
            self.save("{}~".format(arguments.output))
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

    def _generate_dataset(self, size, n, c, p, **kwargs):
        return ds.Synthetic(m=size, n=n, c=c, p=p, random_state=self.state)


class GmonksRunner(BaseRunner):
    """Batch run train/test with the gmonks dataset."""

    def _generate_dataset(self, size, d, **kwargs):
        return ds.GMonks(m=size, d=d, random_state=self.state)


# class WebkbRunner(BaseRunner):

#     def _single_run(self, arguments):
#         """Generate a dataset an train/test all the kernels on it."""
#         if arguments.verbose:
#             print("#{} {}".format(self.state, time.asctime()))
#         # Use the appropiate dataset:
#         dataset, data_arguments = self._generate_dataset(arguments=arguments)
#         X, y, categorize = dataset
#         # Split the data in train and test:
#         print(X.shape, y.shape)
#         X_train, X_test, y_train, y_test = cv.train_test_split(X, y,
#             train_size=arguments.train_size,
#             test_size=arguments.test_size,
#             random_state=self.state,
#             )
#         # Compute other stuff needed for some kernels:
#         Xcat_train = categorize(X_train)
#         Xcat_test = categorize(X_test)
#         pgen = get_pgen(X_train)
#         # Specify the cross-validation to use:
#         cvf = cv.StratifiedKFold(y_train, arguments.folds)
#         # Test preformance with every kernel:
#         kernels = {}
#         if arguments.verbose:
#             print('Running rbf...')
#         kernels['rbf'] = models.RBF.evaluate(
#             cvf, X_train, X_test, y_train, y_test)
#         if arguments.verbose:
#             print('Running k0...')
#         kernels['k0'] = models.K0.evaluate(
#             cvf, Xcat_train, Xcat_test, y_train, y_test)
#         if arguments.verbose:
#             print('Running k1...')
#         kernels['k1'] = models.K1.evaluate(
#             cvf, Xcat_train, Xcat_test, y_train, y_test, pgen=pgen)
#         if arguments.verbose:
#             print('Running k2...')
#         kernels['k2'] = models.K2.evaluate(
#             cvf, Xcat_train, Xcat_test, y_train, y_test, pgen=pgen)
#         if arguments.verbose:
#             print('Running m1...')
#         kernels['m1'] = models.M1.evaluate(
#             cvf, Xcat_train, Xcat_test, y_train, y_test, pgen=pgen)
#         if arguments.verbose:
#             print('Running elk...')
#         kernels['elk'] = models.ELK.evaluate(
#             cvf, X_train, X_test, y_train, y_test)
#         # Update stuff and return results:
#         self.state += 1
#         self.results.append({
#             'timestamp': time.asctime(),
#             'run_arguments': dict(arguments._get_kwarguments()),
#             'data_arguments': data_arguments,
#             'kernels': kernels,
#             })
#         # Save partial results when specified:
#         if arguments.tmp:
#             self.save("{}~".format(arguments.output))


#     def _generate_dataset(self, arguments):
#         return datasets.webkb(), {}

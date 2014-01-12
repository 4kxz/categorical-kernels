import time

import numpy as np
from sklearn import cross_validation as cv

from kcat import datasets
from kcat.kernels import train_test
from kcat.kernels.utils import get_pgen


rbf = train_test.TrainTestRBF()
k0 = train_test.TrainTestK0()
k1 = train_test.TrainTestK1()
k2 = train_test.TrainTestK2()


class BatchRunner:

    def __init__(self, state):
        self.state = state

    def generate_dataset(self, args):
        raise NotImplementedError

    def single_run(self, args):
        print(time.asctime())
        # Use the appropiate dataset:
        dataset, data_args = self.generate_dataset(args=args)
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
        print('Training rbf...')
        kernels['rbf'] = rbf.train_test(cvf, Xb_train, y_train, Xb_test, y_test)
        print('Training k0...')
        kernels['k0'] = k0.train_test(cvf, X_train, y_train, X_test, y_test)
        print('Training k1...')
        kernels['k1'] = k1.train_test(cvf, X_train, y_train, X_test, y_test, pgen=pgen)
        print('Training k2...')
        kernels['k2'] = k2.train_test(cvf, X_train, y_train, X_test, y_test, pgen=pgen)
        # Update stuff and return results:
        self.state += 1
        return {
            'timestamp': time.asctime(),
            'run_args': dict(args._get_kwargs()),
            'data_args': data_args,
            'kernels': kernels,
            }

    def batch_run(self, args):
        return [self.single_run(args) for i in range(args.iterations)]

    def run(self, args):
        return self.batch_run(args)


class SyntheticRunner(BatchRunner):

    def generate_dataset(self, args):
        data_args = {
            'm': args.test_size + args.train_size,
            'n': args.attributes,
            'c': args.classes,
            'p': args.parameter,
            'random_state': self.state,
        }
        return datasets.synthetic(**data_args), data_args

    def run(self, args):
        if args.p_range:
            return self.batch_run_p_range(args)
        else:
            return self.batch_run(args)

    def batch_run_p_range(self, args):
        results = []
        for i in range(args.iterations):
            for p in np.arange(0.4, 0.6, 0.1):
                args.parameter = p;
                result = self.single_run(args)
                results.append(result)
        return results


class GmonksRunner(BatchRunner):

    def generate_dataset(self, args):
        data_args = {
            'm': args.test_size + args.train_size,
            'n': args.attributes,
            'c': args.classes,
            'p': args.parameter,
            'random_state': self.state,
        }
        return datasets.synthetic(**data_args), data_args

#!/usr/bin/env python

import argparse
import json
import time

from sklearn import cross_validation as cv

from kcat.datasets import gmonks, synthetic
from kcat.kernels.train_test import (
    train_rbf, train_k0, train_k1, train_k2,
    test_rbf, test_k0, test_k1, test_k2,
)
from kcat.kernels.utils import get_pgen


if __name__ == '__main__':
    # Input handling:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        default='synthetic',
    )
    parser.add_argument(
        '-m', '--size',
        default='300',
    )
    parser.add_argument(
        '-i', '--iterations',
        default='1',
    )
    args = parser.parse_args()
    size = int(args.size)
    iterations = int(args.iterations)
    # Go!
    for i in range(iterations):
        # Load previous data:
        with open("results.json", "r") as f:
            data = json.load(f)
        # Get next id:
        lastid = -1
        for d in data:
            lastid = d['id'] if lastid < d['id'] else lastid
        nextid = lastid + 1
        results = {'id': nextid, 'timestamp': time.asctime()}
        print(results['timestamp'])
        # Generate new dataset:
        if args.dataset == 'gmonks':
            data_args = dict(m=size, d=1, random_state=nextid)
            X, y, bincoder = gmonks(**data_args)
        elif args.dataset == 'synthetic':
            data_args = dict(m=size, n=25, c=4, p=0.5, random_state=nextid)
            X, y, bincoder = synthetic(**data_args)
        else:
            raise ValueError("Invalid dataset.")
        # Save the args:
        print(data_args)
        results['data_args'] = data_args
        # Split the data in train and test:
        split = cv.train_test_split(X, y, test_size=2/3, random_state=nextid)
        X_train, X_test, y_train, y_test = split
        # Compute stuff needed for some kernels:
        Xb_train =  bincoder(X_train)
        Xb_test = bincoder(X_test)
        pgen = get_pgen(X_train)
        # Specify the cross-validation to use:
        cvf = cv.StratifiedKFold(y_train, 5)
        # Train the classifiers:
        print('Training rbf...')
        rbf = train_rbf(Xb_train, y_train, cvf)
        print('Training k0...')
        k0 = train_k0(X_train, y_train, cvf)
        print('Training k1...')
        k1 = train_k1(X_train, y_train, pgen, cvf)
        print('Training k2...')
        k2 = train_k2(X_train, y_train, pgen, cvf)
        # Test preformance:
        results['kernels'] = {}
        results['kernels']['rbf'] = {
            'params': rbf.params,
            'score': test_rbf(rbf.estimator, Xb_test, y_test),
        }
        results['kernels']['k0'] = {
            'params': k0.params,
            'score': test_k0(k0.estimator, X_train, X_test, y_test, k0.params[1]),
        }
        results['kernels']['k1'] = {
            'params': k1.params,
            'score': test_k1(k1.estimator, X_train, X_test, y_test, pgen, k1.params[1]),
        }
        results['kernels']['k2'] = {
            'params': k1.params,
            'score': test_k2(k2.estimator, X_train, X_test, y_test, pgen, k2.params[1]),
        }
        # Save results:
        data.append(results)
        with open("results.json", "w+") as f:
            f.write(json.dumps(data))

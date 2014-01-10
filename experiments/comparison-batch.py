#!/usr/bin/env python

import argparse
import json
import time

from sklearn import cross_validation as cv

from kcat.datasets import gmonks, synthetic
from kcat.kernels import train_test as tt
from kcat.kernels.utils import get_pgen

rbf = tt.TrainTestRBF()
k0 = tt.TrainTestK0()
k1 = tt.TrainTestK1()
k2 = tt.TrainTestK2()

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
    parser.add_argument(
        '-p', '--param',
        action='store_true',
        )
    args = parser.parse_args()
    m = int(args.size)
    # Go!
    for i in range(int(args.iterations)):
        # Load previous data:
        with open("comparison-results.json", "r") as f:
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
            data_args = dict(m=m, d=1, random_state=nextid)
            X, y, bincoder = gmonks(**data_args)
        elif args.dataset == 'synthetic':
            p = (nextid % 9 + 1) * 0.1 if args.param else 0.5
            data_args = dict(m=m, n=25, c=4, p=p, random_state=nextid)
            X, y, bincoder = synthetic(**data_args)
        else:
            raise ValueError("Invalid dataset.")
        # Save the args:
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
        # Test preformance:
        results['kernels'] = {}
        print('Training rbf...')
        results['kernels']['rbf'] = rbf.train_test(cvf, Xb_train, y_train, Xb_test, y_test)
        print('Training k0...')
        results['kernels']['k0'] = k0.train_test(cvf, X_train, y_train, X_test, y_test)
        print('Training k1...')
        results['kernels']['k1'] = k1.train_test(cvf, X_train, y_train, X_test, y_test, pgen=pgen)
        print('Training k2...')
        results['kernels']['k2'] = k2.train_test(cvf, X_train, y_train, X_test, y_test, pgen=pgen)
        # Save results:
        data.append(results)
        with open("comparison-results.json", "w+") as f:
            f.write(json.dumps(data))

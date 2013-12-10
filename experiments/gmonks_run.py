#!/usr/bin/env python

import json
import time

from sklearn import cross_validation as cv

from kcat.datasets import gmonks
from kcat.kernels.train_test import (
    train_rbf, train_k0, train_k1, train_k2,
    test_rbf, test_k0, test_k1, test_k2,
)
from kcat.kernels.utils import get_pgen


#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

with open("results.json", "r") as f:
    data = json.load(f)

# Get next id:
lastid = -1
for d in data:
    lastid = d['id'] if lastid < d['id'] else lastid
nextid = lastid + 1


#------------------------------------------------------------------------------
# Generate dataset
#------------------------------------------------------------------------------

data_args = dict(m=300, d=1, random_state=nextid)
X, y, bincoder = gmonks(**data_args)
# Split the data in train and test:
split = cv.train_test_split(X, y, test_size=2/3, random_state=nextid)
X_train, X_test, y_train, y_test = split
# Compute stuff needed for some kernels:
Xb_train =  bincoder(X_train)
Xb_test = bincoder(X_test)
pgen = get_pgen(X_train)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

new_data = {
    'id': nextid,
    'data_args': data_args,
    'timestamp': time.asctime(),
    'kernels': {},
}

print(new_data)

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
new_data['kernels']['rbf'] = {
    'params': rbf.params,
    'score': test_rbf(rbf.estimator, Xb_test, y_test),
}
new_data['kernels']['k0'] = {
    'params': k0.params,
    'score': test_k0(k0.estimator, X_train, X_test, y_test, k0.params[1]),
}
new_data['kernels']['k1'] = {
    'params': k1.params,
    'score': test_k1(k1.estimator, X_train, X_test, y_test, pgen, k1.params[1]),
}
new_data['kernels']['k2'] = {
    'params': k1.params,
    'score': test_k2(k2.estimator, X_train, X_test, y_test, pgen),
}

#------------------------------------------------------------------------------
# Save data
#------------------------------------------------------------------------------

data.append(new_data)
with open("results.json", "w+") as f:
    f.write(json.dumps(data))

#!/usr/bin/env python
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KERNELS = ('rbf', 'k0', 'k1', 'k2')
COLORS = ('k', 'r', 'g', 'b')


with open("comparison-results.json", "r") as f:
    # Flatten data before loading into pandas.
    items = []
    for raw in json.load(f):
        for k in raw['kernels']:
            item = {
                'id': raw['id'],
                'timestamp': raw['timestamp'],
                'kernel': k,
                'train_score': raw['kernels'][k]['train_score'],
                'test_score': raw['kernels'][k]['test_score'],
                }
            item.update(raw['data_args'])
            item.update(raw['kernels'][k]['best_parameters'])
            items.append(item)

data = pd.DataFrame(items)

plt.figure()
data.groupby('kernel')['train_score', 'test_score'].mean().plot()
plt.savefig('test.png')
print(data.groupby(('kernel', 'p')).mean())

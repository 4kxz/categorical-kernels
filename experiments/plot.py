#!/usr/bin/env python

import json
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KERNELS = ('rbf', 'k0', 'k1', 'k2')
COLORS = ('k', 'r', 'g', 'b')


parser = ArgumentParser()
parser.add_argument(
    'filename',
    metavar='FILENAME',
    type=str,
    help='file to read',
    )

args = parser.parse_args()

with open(args.filename, "r") as f:
    # Flatten data before loading into pandas.
    items = []
    for raw in json.load(f):
        for k in raw['kernels']:
            item = {
                'kernel': k,
                'train_score': raw['kernels'][k]['train_score'],
                'test_score': raw['kernels'][k]['test_score'],
                }
            item.update(raw['kernels'][k]['best_parameters'])
            item.update(raw['run_args'])
            item.update(raw['data_args'])
            items.append(item)

# Create dataframe and clean things up
df = pd.DataFrame(items)
df['p'] = df['p'].round(decimals=2)

# Plot scores for each kernel
by_kernel = df.groupby('kernel').groups
for k in by_kernel:
    fig = plt.figure()
    df.loc[by_kernel[k], ['p', 'train_score', 'test_score']].boxplot(by='p')
    plt.savefig('test{}.png'.format(k))

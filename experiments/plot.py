#!/usr/bin/env python

import json
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from mpltools import style
    style.use('ggplot')
except:
    pass

KERNELS = ('rbf', 'k0', 'k1', 'k2')
COLORS = ('k', 'r', 'g', 'b')


parser = ArgumentParser()
parser.add_argument(
    'filename',
    metavar='FILENAME',
    type=str,
    help='file to read',
    )
parser.add_argument(
    '-o', '--output',
    default='plot',
    type=str,
    help='filename where the output should be saved',
    )
parser.add_argument(
    '-s', '--synthetic',
    action='store_true',
    )
parser.add_argument(
    '-g', '--gmonks',
    action='store_true',
    )

args = parser.parse_args()

with open(args.filename, "r") as f:
    # Flatten data before loading into pandas.
    items = []
    for raw in json.load(f):
        for k in raw['kernels']:
            item = {
                'kernel': k,
                'train_error': 1 - raw['kernels'][k]['train_score'],
                'test_error': 1 - raw['kernels'][k]['test_score'],
                }
            item.update(raw['kernels'][k]['best_parameters'])
            item.update(raw['run_args'])
            item.update(raw['data_args'])
            items.append(item)

# Create dataframe and clean things up
df = pd.DataFrame(items)

if args.synthetic:
    # Plot scores for each kernel grouped by p
    df['p'] = df['p'].round(decimals=2)
    by_kernel = df.groupby('kernel').groups
    for k in by_kernel:
        fig = plt.figure()
        df.loc[by_kernel[k], ['p', 'train_error', 'test_error']].boxplot(by='p')
        plt.savefig('{}-p-error-{}.png'.format(args.output, k))

fig = plt.figure()
df.loc[:, ['kernel', 'train_error', 'test_error']].boxplot(by='kernel')
plt.savefig('{}-train-test.png'.format(args.output))

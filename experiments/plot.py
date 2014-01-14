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
    help="json file containg execution data",
    )
parser.add_argument(
    '-o', '--output',
    default='plot',
    type=str,
    help="directory or prefix used to save the images",
    )
parser.add_argument(
    '-k', '--kernel',
    default=None,
    help="filter results and plot only those with a specific kernel",
    )
parser.add_argument(
    '-d', '--dataset',
    default=None,
    help="filter results and plot only those with a specific dataset",
    )
parser.add_argument(
    '-g', '--group-by',
    default='kernel',
    help="plot results grouped by the passed attribute",
    )
parser.add_argument(
    '--synthetic',
    action='store_true',
    help="plot graphs for synthetic dataset",
    )
parser.add_argument(
    '--gmonks',
    action='store_true',
    help="plot graphs for gmonks dataset",
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

# args.kernel filters a single kernel:
if args.kernel is not None:
    df = df[df.kernel == args.kernel]

fig = plt.figure()
df.loc[:, [args.group_by, 'train_error', 'test_error']].boxplot(by=args.group_by)
plt.savefig('{}-train-test.png'.format(args.output))
fig = plt.figure()
df.loc[:, [args.group_by]].groupby(args.group_by).count().plot(kind='bar')
plt.savefig('{}-count.png'.format(args.output))

if args.synthetic:
    df = df[df.dataset == 'synthetic']
    # Plot scores for each kernel grouped by p
    df['p'] = df['p'].round(decimals=2)
    by_kernel = df.groupby('kernel').groups
    for k in by_kernel:
        fig = plt.figure()
        df.loc[by_kernel[k], ['p', 'train_error', 'test_error']].boxplot(by='p')
        plt.savefig('{}-p-error-{}.png'.format(args.output, k))

if args.gmonks:
    pass

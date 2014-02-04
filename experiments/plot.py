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
        for k in raw['evaluation']:
            item = {
                'kernel': k,
                'train_error': 1 - raw['evaluation'][k]['train_score'],
                'test_error': 1 - raw['evaluation'][k]['test_score'],
                }
            item.update(raw['evaluation'][k]['best_parameters'])
            item.update(raw['arguments'])
            items.append(item)

# Create dataframe and clean things up.
df = pd.DataFrame(items)

# Basic filters.
if args.kernel is not None:
    df = df[df.kernel == args.kernel]
if args.dataset is not None:
    df = df[df.dataset == args.dataset]

# Plot train and test error.
fig = plt.figure()
groups = [args.group_by, 'train_error', 'test_error']
df.loc[:, groups].boxplot(by=args.group_by)
plt.ylim(0, 1)
plt.savefig('{}-train-test.png'.format(args.output))

# Print count for each group.
fig = plt.figure()
df.loc[:, [args.group_by]].groupby(args.group_by).count().plot(kind='bar')
plt.savefig('{}-count.png'.format(args.output))

if args.synthetic:
    df = df[df.dataset == 'Synthetic']
    # Plot scores for each kernel grouped by p
    df['p'] = df['p'].round(decimals=2)
    by_kernel = df.groupby('kernel').groups
    for k in by_kernel:
        fig = plt.figure()
        groups = ['p', 'train_error', 'test_error']
        df.loc[by_kernel[k], groups].boxplot(by='p')
        plt.ylim(0, 1)
        plt.savefig('{}-p-error-{}.png'.format(args.output, k))

if args.gmonks:
    pass

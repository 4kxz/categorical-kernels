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

data = pd.DataFrame(items)

groups = ['kernel', 'p']
columns = ['train_score', 'test_score']
print(data.groupby(groups).mean()[columns])

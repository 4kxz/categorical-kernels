#!/usr/bin/env python

import json

from pylab import *

KERNELS = ('rbf', 'k0', 'k1', 'k2')

with open("experiments/results/synthetic.json", "r") as f:
    scores = [[] for k in KERNELS]
    for line in json.load(f):
        for i, k in enumerate(KERNELS):
            scores[i].append(line['kernels'][k]['score'])

    figure()
    boxplot(scores)
    xticks(range(1, 5), KERNELS)
    savefig('test.png')

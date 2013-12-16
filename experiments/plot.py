#!/usr/bin/env python

import json

from pylab import *

KERNELS = ('rbf', 'k0', 'k1', 'k2')

with open("results.json", "r") as f:
    scores = [[] for k in KERNELS]
    for line in json.load(f):
        for i, k in enumerate(KERNELS):
            scores[i].append(line['kernels'][k]['score'])
    figure()
    boxplot(scores)
    xticks(range(1, 5), KERNELS)
    savefig('plota.png')
    figure()
    plot(scores, '+b', alpha=0.1)
    xticks(range(0, 4), KERNELS)
    xlim(-.5, 3.5)
    savefig('plotb.png')

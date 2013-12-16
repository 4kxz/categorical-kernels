#!/usr/bin/env python
import json

from pylab import *

KERNELS = ('rbf', 'k0', 'k1', 'k2')

with open("results.json", "r") as f:
    scores = [[] for k in KERNELS]
    for line in json.load(f):
        for i, k in enumerate(KERNELS):
            if int(line['data_args']['p'] * 10) == 5:
                scores[i].append(line['kernels'][k]['score'])
    figure()
    bp = boxplot(scores)
    setp(bp['boxes'], color='black')
    setp(bp['whiskers'], color='black')
    setp(bp['fliers'], color='black')
    setp(bp['medians'], color='black')
    xticks(range(1, 5), KERNELS)
    savefig('plot.png')
with open("results.json", "r") as f:
    scores = [[[] for _ in range(0,9)] for _ in KERNELS]
    for line in json.load(f):
        for i, k in enumerate(KERNELS):
            p = int(line['data_args']['p'] * 10) - 1
            scores[i][p].append(line['kernels'][k]['score'])
    colors = ('k', 'r', 'g', 'b')
    for i, k in enumerate(KERNELS):
        figure()
        S = np.array(scores[i])
        Sm = S.mean(axis=1)
        Ss = S.std(axis=1)
        plot(np.arange(1, 10), Sm, c=colors[i], alpha=.2)
        fill(np.concatenate([np.arange(1, 10), np.arange(1, 10)[::-1]]),
             np.concatenate([Sm + Ss, (Sm - Ss)[::-1]]),
             alpha=.1, fc=colors[i], ec='None', label='1 stdev'
             )
        xticks(np.arange(1, 10), np.arange(1, 10) * 0.1)
        ylim(0.25, 1)
        bp = boxplot(scores[i])
        setp(bp['boxes'], color='black')
        setp(bp['whiskers'], color='black')
        setp(bp['fliers'], color=colors[i], marker='x')
        setp(bp['medians'], color=colors[i])
        xticks(np.arange(1, 10), np.arange(1, 10) * 0.1)
        savefig('plot-{}.png'.format(KERNELS[i]))
    figure()
    for i, k in enumerate(KERNELS):
        S = np.array(scores[i])
        Sm = S.mean(axis=1)
        Ss = S.std(axis=1)
        plot(np.arange(1, 10), Sm, c=colors[i], alpha=.75)
        fill(np.concatenate([np.arange(1, 10), np.arange(1, 10)[::-1]]),
             np.concatenate([Sm + Ss, (Sm - Ss)[::-1]]),
             alpha=.1, fc=colors[i], ec='None', label='1 stdev'
             )
        xticks(np.arange(1, 10), np.arange(1, 10) * 0.1)
        ylim(0.25, 1)
    savefig('plot-all.png')

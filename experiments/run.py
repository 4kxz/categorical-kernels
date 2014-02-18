#!/usr/bin/env python

from argparse import ArgumentParser

import runners


if __name__ == '__main__':
    # Common arguments
    parser = ArgumentParser(prog='prog')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='show progress',
        )
    parser.add_argument(
        '-o', '--output',
        default='results.json',
        type=str,
        help='filename where the output should be saved',
        )
    parser.add_argument(
        '-i', '--iterations',
        default=1,
        type=int,
        help='number of repetitions',
        )
    parser.add_argument(
        '-s', '--random-state',
        default=0,
        type=int,
        help='seed for the random state',
        )
    parser.add_argument(
        '-r', '--train-size',
        default=100,
        type=int,
        help="number of elements in the train set",
        )
    parser.add_argument(
        '-e', '--test-size',
        default=200,
        type=int,
        help="number of elements in the test set",
        )
    parser.add_argument(
        '-f', '--folds',
        default=5,
        type=int,
        help="number of folds to use in cross-validation",
        )
    subparsers = parser.add_subparsers(dest='dataset')
    # Synthetic dataset arguments
    synthetic = subparsers.add_parser('Synthetic')
    synthetic.add_argument(
        '-n', '--n',
        default=25,
        type=int,
        help="number of attributes for each example",
        )
    synthetic.add_argument(
        '-c', '--c',
        default=4,
        type=int,
        help="number of classes",
        )
    synthetic.add_argument(
        '-p', '--p',
        default=0.5,
        type=float,
        help="adjust frequency of random values",
        )
    synthetic.add_argument(
        '--p-range',
        action='store_true',
        help="try different values for p in increments of 0.1",
        )
    # GMonks dataset arguments
    gmonks = subparsers.add_parser('GMonks')
    gmonks.add_argument(
        '-d', '--d',
        default=1,
        type=int,
        help="number of attributes",
        )
    # WebKB dataset arguments
    webkb = subparsers.add_parser('WebKB')
    # Parse arguments
    args = parser.parse_args()
    # Load appropiate runner by name
    runner = getattr(runners, args.dataset + 'Runner')
    # Run and save results to file
    tester = runner(args.random_state, args.verbose)
    filename = args.output
    args = dict(args._get_kwargs())
    # Remove unnecessary values
    del args['output'], args['random_state'], args['verbose']
    # Execution proper
    tester.run(**args)
    tester.save(filename)

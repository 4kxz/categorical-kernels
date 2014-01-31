#!/usr/bin/env python

import sys

import parsers
import runners

if __name__ == '__main__':
    try:
        # Get the dataset name first
        dataset = sys.argv[1]
        # The parser and runner are loaded by name
        parser = getattr(parsers, dataset + 'Parser')
        runner = getattr(runners, dataset + 'Runner')
    except AttributeError:
        # This happens when the classes for are not implemented
        print("Invalid dataset {}".format(dataset))
    else:
        # Run and save results to file
        args = parser().parse_args()
        filename = args.output
        tester = runner(args.seed, args.verbose)
        kwargs = dict(args._get_kwargs())
        del kwargs['output']
        del kwargs['seed']
        del kwargs['verbose']
        tester.run(**kwargs)
        tester.save(filename)

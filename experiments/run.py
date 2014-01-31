#!/usr/bin/env python

import sys

import parsers
import runners

if __name__ == '__main__':
    try:
        # Get the dataset name first
        dataset = sys.argv[1].title()
        # The parser and runner are loaded by name
        parser_init = getattr(parsers, dataset + 'Parser')
        runner_init = getattr(runners, dataset + 'Runner')
    except AttributeError:
        # This happens when the classes for are not implemented
        print("Invalid dataset {}".format(dataset))
    else:
        # Run and save results to file
        args = parser_init().parse_args()
        tester = runner_init(args.random_state, args.tmp, args.verbose)
        kwargs = dict(args._get_kwargs())
        tester.run(**kwargs)
        tester.save(args.output)

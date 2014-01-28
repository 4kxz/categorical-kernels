#!/usr/bin/env python

import sys

import parsers
import runners

if __name__ == '__main__':
    try:
        # Get the dataset name first
        dataset = sys.argv[1].title()
        # The parser and runner are loaded by name
        parser_class = getattr(parsers, dataset + 'Parser')
        runner_class = getattr(runners, dataset + 'Runner')
    except AttributeError:
        # This happens when the classes for are not implemented
        print("Invalid dataset {}".format(dataset))
    else:
        # Run and save results to file
        args = parser_class().parse_args()
        tester = runner_class(args.random_state)
        tester.run(args)
        tester.save(args.output)

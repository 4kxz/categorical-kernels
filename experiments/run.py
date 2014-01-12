#!/usr/bin/env python

import json

import parsing
import running

if __name__ == '__main__':
    try:
        # Get the dataset name first
        dataset = parsing.BaseParser().parse_args().dataset.title()
        # The parser and runner are loaded by name
        parser = getattr(parsing, dataset + 'Parser')()
        runner = getattr(running, dataset + 'Runner')(0)
    except AttributeError:
        # This happens when the classes for are not implemented
        print("Invalid dataset {}".format(dataset))
    else:
        # Run and save results to file
        args = parser.parse_args()
        results = runner.run(args)
        with open(args.output, "w+") as f:
            f.write(json.dumps(results))

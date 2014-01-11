#!/usr/bin/env python

import json

import parsing
import running

if __name__ == '__main__':
    parser = parsing.SyntheticParser()
    runner = running.SyntheticRunner(0)
    args = parser.parse_args()
    results = runner.run(args)
    with open("results-batch.json", "w+") as f:
        f.write(json.dumps(results))

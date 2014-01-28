from argparse import ArgumentParser


class BaseParser(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            'dataset',
            metavar='DATASET',
            type=str,
            help='a valid dataset name',
            )
        self.add_argument(
            '-v', '--verbose',
            action='store_true',
            help='show progress',
            )
        self.add_argument(
            '-t', '--tmp',
            action='store_true',
            help='store partial results in temporary file',
            )
        self.add_argument(
            '-o', '--output',
            default='results.json',
            type=str,
            help='filename where the output should be saved',
            )
        self.add_argument(
            '-i', '--iterations',
            default=1,
            type=int,
            help='number of repetitions',
            )
        self.add_argument(
            '-s', '--random-state',
            default=0,
            type=int,
            help='seed for the random state',
            )
        self.add_argument(
            '-r', '--train-size',
            default=100,
            type=int,
            )
        self.add_argument(
            '-e', '--test-size',
            default=200,
            type=int,
            )
        self.add_argument(
            '-f', '--folds',
            default=5,
            type=int,
            )


class SyntheticParser(BaseParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            '-n', '--n',
            default=25,
            type=int,
            )
        self.add_argument(
            '-c', '--c',
            default=4,
            type=int,
            )
        self.add_argument(
            '-p', '--p',
            default=0.5,
            type=float,
            )
        self.add_argument(
            '--p-range',
            action='store_true',
            )


class GmonksParser(BaseParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            '-d', '--d',
            default=1,
            type=int,
            )


class WebkbParser(BaseParser):
    pass

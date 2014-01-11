from argparse import ArgumentParser


class BaseParser(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            '-m', '--size',
            default=300,
            type=int,
            )
        self.add_argument(
            '-s', '--split',
            default=2/3,
            type=float,
            )
        self.add_argument(
            '-f', '--folds',
            default=5,
            type=int,
            )
        self.add_argument(
            '-i', '--iterations',
            default=1,
            type=int,
            )


class SyntheticParser(BaseParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            '-n', '--attributes',
            default=25,
            type=int,
            )
        self.add_argument(
            '-c', '--classes',
            default=4,
            type=int,
            )
        self.add_argument(
            '-p', '--parameter',
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
            '-n', '--attributes',
            default=1,
            type=int,
            )

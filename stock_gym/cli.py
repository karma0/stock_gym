# -*- coding: utf-8 -*-

"""Console script for stock_gym."""
import sys
import click
from stock_gym import stock_gym


@click.command()
def main(args=None):
    """Console script for stock_gym."""
    stock_gym.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

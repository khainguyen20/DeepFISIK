"""Convenience launcher for the classification training command.

The script prepends the ``classification`` subcommand and delegates the rest
of the arguments to :func:`deepfisik.cli.main`.
"""

from __future__ import annotations

import sys

from deepfisik.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["classification", *sys.argv[1:]]))

"""Convenience launcher for the tetramerss regression training command.

The script prepends the ``interactions`` subcommand and selects the tetramer
model while reusing the shared CLI implementation.
"""

from __future__ import annotations

import sys

from deepfisik.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["interactions", "--model", "tetramer", *sys.argv[1:]]))

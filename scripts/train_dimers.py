"""Convenience launcher for the dimers regression training command.

The script prepends the ``interactions`` subcommand and selects the dimer
model so the user can launch the right regression pipeline with one file.
"""

from __future__ import annotations

import sys

from deepfisik.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["interactions", "--model", "dimer", *sys.argv[1:]]))

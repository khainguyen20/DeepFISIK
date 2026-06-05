"""Convenience launcher for the trimerss regression training command.

The script prepends the ``interactions`` subcommand and selects the trimer
model so the training entry point stays consistent across tasks.
"""

from __future__ import annotations

import sys

from deepfisik.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["interactions", "--model", "trimer", *sys.argv[1:]]))

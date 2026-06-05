"""Convenience launcher for dimers regression inference.

This script forwards to :func:`deepfisik.inference.core.main_interactions`
and hard-codes ``--model dimer`` so the dimers checkpoint can be evaluated with
minimal typing.
"""

from __future__ import annotations

import sys

from deepfisik.inference.core import main_interactions


if __name__ == "__main__":
    raise SystemExit(main_interactions(["--model", "dimer", *sys.argv[1:]]))

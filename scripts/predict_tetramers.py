"""Convenience launcher for tetramers regression inference.

This script forwards to :func:`deepfisik.inference.core.main_interactions`
and hard-codes ``--model tetramers`` so the tetramers checkpoint can be
evaluated with minimal typing.
"""

from __future__ import annotations

import sys

from deepfisik.inference.core import main_interactions


if __name__ == '__main__':
    raise SystemExit(main_interactions(["--model", "tetramer", *sys.argv[1:]]))

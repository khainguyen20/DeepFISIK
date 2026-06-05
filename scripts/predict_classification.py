"""Convenience launcher for oligo classification inference.

This script forwards to :func:`deepfisik.inference.core.main_oligo` so the
classification inference workflow can be started from ``scripts/`` just like
training.
"""

from deepfisik.inference.core import main_oligo


if __name__ == "__main__":
    raise SystemExit(main_oligo())

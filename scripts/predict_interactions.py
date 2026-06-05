"""Convenience launcher for interaction regression inference.

This script forwards to :func:`deepfisik.inference.core.main_interactions` so
held-out evaluation can be started from ``scripts/``.
"""

from deepfisik.inference.core import main_interactions


if __name__ == "__main__":
    raise SystemExit(main_interactions())

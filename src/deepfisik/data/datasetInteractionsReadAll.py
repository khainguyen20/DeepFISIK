# Copyright (C) 2026, Jaqaman Lab - UTSouthwestern
#
# This file is part of DeepFISIK.
#
# DeepFISIK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepFISIK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepFISIK.  If not, see <http://www.gnu.org/licenses/>.

"""Dataset loader for processed interaction graph files."""

import os

import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import pandas as pd
try:
    from deepfisik.data.graph_generator_interactions_un import *
except ImportError:  # legacy fallback when running the file directly
    from graphGeneratorInteractionsUN import *
import scipy.io
import re

def extract_number(s):
    """Return the numeric suffix from a processed graph filename."""
    m = re.search(r"data_(\d+)\.pt$", s)
    return int(m.group(1)) if m else float("inf")

class SMI(Dataset):
    """PyG dataset for loading processed Deep-FISIK graph samples.

    Parameters
    ----------
    root : str
        Root directory containing the processed dataset files.
    test : bool, optional
        If ``True``, load the test split file pattern.
    transform : callable, optional
        Optional PyG transform applied when returning each sample.
    max_samples : int, optional
        Limit the number of processed files loaded from disk.
    shuffle : bool, optional
        Shuffle the file list before truncating.
    seed : int, optional
        Random seed used when ``shuffle`` is enabled.

    Returns
    -------
    torch_geometric.data.Data
        A processed graph sample loaded from disk.
    """
    def __init__(
        self,
        root,
        test=False,
        transform=None,
        max_samples=None,   # ← NEW
        shuffle=False,      # ← optional
        seed=42             # ← optional
    ):
        """Initialize the dataset and cache the processed file list."""
        self.test = test
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.seed = seed

        super().__init__(root, transform)

        self._files = self._load_file_list()

    def _load_file_list(self):
        """Collect processed files, optionally shuffle, and apply truncation."""
        prefix = "data_test_" if self.test else "data_"
        files = [
            f for f in os.listdir(self.processed_dir)
            if f.startswith(prefix) and f.endswith(".pt")
        ]

        files = sorted(files, key=extract_number)

        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(files)

        if self.max_samples is not None:
            files = files[: self.max_samples]

        return files

    @property
    def raw_file_names(self):
        """Return an empty raw-file list because data are already processed."""
        return []

    @property
    def processed_file_names(self):
        """Return the cached processed file names used by PyG."""
        return self._files

    def len(self):
        """Return the number of processed samples available."""
        return len(self._files)

    def get(self, idx):
        """Load a single processed graph sample from disk."""
        path = os.path.join(self.processed_dir, self._files[idx])
        data = torch.load(path)

        if self.transform:
            data = self.transform(data)

        return data

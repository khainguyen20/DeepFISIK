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

"""Command-line interface for Deep-FISIK training workflows.

The module exposes two main user-facing commands: classification and
interactions. Each command keeps the model construction, data loading, and
training loop inside the package while thin wrapper scripts in ``scripts/``
provide convenience launchers.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, random_split
from torch_geometric.loader import DataLoader, DataListLoader

from deepfisik.data.datasetInteractionsReadAll import SMI
from deepfisik.models.gnn_dimer import GNNDimer
from deepfisik.models.gnn_oligo import GNNOligo
from deepfisik.models.gnn_tetramer import GNNTetramer
from deepfisik.models.gnn_trimer import GNNTrimer
from deepfisik.training.train_interactions_dimers import train_testDimers
from deepfisik.training.train_interactions_trimers import train_testTrimers
from deepfisik.training.train_interactions_tetramers import train_testTetramers
from deepfisik.training.train_oligo import train_testOligo


DATASETS_DIR = Path("Datasets")

DEFAULT_OLIGO_ROOTS = (
    str(DATASETS_DIR / "Images" / "MonomerDataset"),
    str(DATASETS_DIR / "Images" / "DimerDataset"),
    str(DATASETS_DIR / "Images" / "TrimerDataset"),
    str(DATASETS_DIR / "Images" / "TetramerDataset"),
)

DEFAULT_INTERACTION_ROOTS = {
    "dimer": [str(DATASETS_DIR / "PureSimulations" / "DimerDataset")],
    "trimer": [str(DATASETS_DIR / "PureSimulations" / "TrimerDataset")],
    "tetramer": [str(DATASETS_DIR / "PureSimulations" / "TetramerDataset")],
}


def _device(device_name: str | None) -> torch.device:
    """Return the requested device or a sensible CUDA/CPU default."""
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_dirs(base: Path) -> None:
    """Create the output subdirectories used by the training runs."""
    for sub in (
        "parameters",
        "results",
        "intermediateResults",
        "models",
        "checkpoints",
        "finalModel",
    ):
        (base / sub).mkdir(parents=True, exist_ok=True)


def _common_model_frame(**kwargs) -> pd.DataFrame:
    """Build the parameter table saved alongside each experiment."""
    return pd.DataFrame(kwargs)


def _transform_xy(data):
    """Normalize the X/Y node coordinates to the unit interval."""
    x = data.x[:, :3].clone()
    min_x = torch.min(x[:, 0])
    max_x = torch.max(x[:, 0])
    min_y = torch.min(x[:, 1])
    max_y = torch.max(x[:, 1])

    if max_x > min_x:
        x[:, 0] = (x[:, 0] - min_x) / (max_x - min_x)
    if max_y > min_y:
        x[:, 1] = (x[:, 1] - min_y) / (max_y - min_y)

    data.x = x
    return data

OLIGO_META_ATTRS = ["AP2", "AP3", "AP4", "DR2", "DR3", "DR4"]

def strip_oligo_meta(data):
    for attr in OLIGO_META_ATTRS:
        if hasattr(data, attr):
            delattr(data, attr)
    return data


def transform_monomer(data):
    """Prepare a graph sample for the monomer classification label."""
    data = _transform_xy(data)
    data = strip_oligo_meta(data)
    data.oligoLabel = torch.tensor([1, 0, 0, 0]).reshape(1, 4)
    return data


def transform_dimer(data):
    """Prepare a graph sample for the dimer classification label."""
    data = _transform_xy(data)
    data = strip_oligo_meta(data)
    data.oligoLabel = torch.tensor([0, 1, 0, 0]).reshape(1, 4)
    return data


def transform_trimer(data):
    """Prepare a graph sample for the trimer classification label."""
    data = _transform_xy(data)
    data = strip_oligo_meta(data)
    data.oligoLabel = torch.tensor([0, 0, 1, 0]).reshape(1, 4)
    return data


def transform_tetramer(data):
    """Prepare a graph sample for the tetramer classification label."""
    data = _transform_xy(data)
    data = strip_oligo_meta(data)
    data.oligoLabel = torch.tensor([0, 0, 0, 1]).reshape(1, 4)
    return data


def NormalizeCoordinates(data):
    """Normalize node coordinates for the interaction-regression datasets."""
    x = data.x[:, :3].clone()
    min_x = torch.min(x[:, 0])
    max_x = torch.max(x[:, 0])
    min_y = torch.min(x[:, 1])
    max_y = torch.max(x[:, 1])

    if max_x > min_x:
        x[:, 0] = (x[:, 0] - min_x) / (max_x - min_x)
    if max_y > min_y:
        x[:, 1] = (x[:, 1] - min_y) / (max_y - min_y)

    data.x = x
    return data


def _add_oligo_arguments(parser: argparse.ArgumentParser) -> None:
    """Add classification-specific command-line arguments to a parser."""
    parser.add_argument(
        "--dataset-roots",
        nargs=4,
        metavar=("MONO", "DIMER", "TRIMER", "TETRAMER"),
        default=list(DEFAULT_OLIGO_ROOTS),
        help="Paths to the monomer, dimer, trimer, and tetramer datasets.",
    )
    parser.add_argument("--save-dir", default="runs/oligo")
    parser.add_argument("--date", default="run")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument("--layers", type=int, default=7)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=50)


def _add_interactions_arguments(parser: argparse.ArgumentParser) -> None:
    """Add regression-specific command-line arguments to a parser."""
    parser.add_argument(
        "--model",
        choices=("dimer", "trimer", "tetramer"),
        default="tetramer",
        help="Which interaction model to train.",
    )
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        default=None,
        metavar="ROOT",
        help=(
            "One or more paths to interaction datasets. "
            "If omitted, the default dataset for the selected model is used. "
            "Multiple roots are concatenated."
        ),
    )
    parser.add_argument("--save-dir", default="runs/interactions")
    parser.add_argument("--date", default="run")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument("--layers", type=int, default=7)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=50)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level Deep-FISIK CLI parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Deep-FISIK command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    classification = subparsers.add_parser(
        "classification",
        help="Train the oligo classification model.",
    )
    _add_oligo_arguments(classification)
    classification.set_defaults(handler=run_oligo)

    interactions = subparsers.add_parser(
        "interactions",
        help="Train one of the interaction regression models.",
    )
    _add_interactions_arguments(interactions)
    interactions.set_defaults(handler=run_interactions)

    return parser


def _resolve_interaction_roots(args: argparse.Namespace) -> list[str]:
    """Return the default interaction roots for the selected model."""
    if args.dataset_roots is not None:
        return list(args.dataset_roots)
    return list(DEFAULT_INTERACTION_ROOTS[args.model])


def run_oligo(args: argparse.Namespace):
    """Run the oligo classification training pipeline."""
    device = _device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_roots = [Path(p) for p in args.dataset_roots]
    transforms = [
        transform_monomer,
        transform_dimer,
        transform_trimer,
        transform_tetramer,
    ]

    datasets = [
        SMI(str(root), transform=transform)
        for root, transform in zip(dataset_roots, transforms)
    ]
    dataset = ConcatDataset(datasets)

    generator = torch.Generator().manual_seed(args.seed)
    train_len = int(math.ceil(0.8 * len(dataset)))
    test_len = len(dataset) - train_len
    train_dataset, test_dataset = random_split(
        dataset,
        [train_len, test_len],
        generator=generator,
    )

    save_path = Path(args.save_dir)
    _ensure_dirs(save_path)

    model_params = _common_model_frame(
        node_model_embedding_size=[args.embedding_dim],
        edge_model_embedding_size=[args.embedding_dim],
        model_attention_heads=[6],
        model_attention_head_dimension=[args.embedding_dim],
        model_layers=[args.layers],
        model_attention_dropout_rate=[0],
        model_top_k_ratio=[1.0],
        model_top_k_every_n=[10],
        model_edge_dim=train_dataset[0].edge_attr.shape[1],
        model_feature_size=train_dataset[0].x.shape[1],
        model_batch_size=args.batch_size,
        model_layer_dropout_rate=[0],
        model_laplacian=[10],
        model_noClasses=4,
        Epochs=args.epochs,
        E_Patience=args.patience,
        Learning_Rate=args.learning_rate,
        Weight_Decay=args.weight_decay,
        Time_Step=args.time_step,
        Length_Dataset=len(dataset),
    )
    model_params.to_csv(
        save_path / "parameters" / f"Parameter_{args.date}.csv",
        index=False,
    )

    if torch.cuda.device_count() > 1:
        train_loader = DataListLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_loader = DataListLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )

    model, results = train_testOligo(
        device,
        train_loader,
        test_loader,
        GNNOligo,
        args.time_step,
        model_params,
        str(save_path),
        args.date,
        checkpointFlag=False,
        learning_rate=args.learning_rate,
        e_patience=args.patience,
        n_epochs=args.epochs,
        weight_decay=args.weight_decay,
    )

    results.to_csv(save_path / "results" / f"Results_{args.date}.csv", index=False)
    torch.save(model, save_path / "finalModel" / f"modelGNN_{args.date}.pt")
    return model, results


def run_interactions(args: argparse.Namespace):
    """Run the interaction regression pipeline for the selected model."""
    device = _device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_registry = {
        "dimer": {
            "model": GNNDimer,
            "trainer": train_testDimers,
            "label": "dimer",
        },
        "trimer": {
            "model": GNNTrimer,
            "trainer": train_testTrimers,
            "label": "trimer",
        },
        "tetramer": {
            "model": GNNTetramer,
            "trainer": train_testTetramers,
            "label": "tetramer",
        },
    }

    selected = model_registry[args.model]
    dataset_roots = [Path(root) for root in args.dataset_roots]
    datasets = [SMI(str(root), transform=NormalizeCoordinates) for root in dataset_roots]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    generator = torch.Generator().manual_seed(args.seed)
    train_len = int(math.ceil(0.8 * len(dataset)))
    test_len = len(dataset) - train_len
    train_dataset, test_dataset = random_split(
        dataset,
        [train_len, test_len],
        generator=generator,
    )

    save_path = Path(args.save_dir) / selected["label"]
    _ensure_dirs(save_path)

    model_params = _common_model_frame(
        node_model_embedding_size=[args.embedding_dim],
        edge_model_embedding_size=[args.embedding_dim],
        model_attention_heads=[6],
        model_attention_head_dimension=[args.embedding_dim],
        model_layers=[args.layers],
        model_attention_dropout_rate=[0],
        model_top_k_ratio=[1.0],
        model_top_k_every_n=[10],
        model_edge_dim=train_dataset[0].edge_attr.shape[1],
        model_feature_size=train_dataset[0].x.shape[1],
        model_batch_size=args.batch_size,
        model_layer_dropout_rate=[0],
        model_laplacian=[10],
        Epochs=args.epochs,
        E_Patience=args.patience,
        Learning_Rate=args.learning_rate,
        Weight_Decay=args.weight_decay,
        Time_Step=args.time_step,
        Length_Dataset=len(dataset),
    )
    model_params.to_csv(
        save_path / "parameters" / f"Parameter_{args.date}.csv",
        index=False,
    )

    if torch.cuda.device_count() > 1:
        train_loader = DataListLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_loader = DataListLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

    model, results = selected["trainer"](
        device,
        train_loader,
        test_loader,
        selected["model"],
        args.time_step,
        model_params,
        str(save_path),
        args.date,
        checkpointFlag=False,
        learning_rate=args.learning_rate,
        e_patience=args.patience,
        n_epochs=args.epochs,
        weight_decay=args.weight_decay,
    )

    results.to_csv(save_path / "results" / f"Results_{args.date}.csv", index=False)
    torch.save(model, save_path / "finalModel" / f"modelGNN_{args.date}.pt")
    return model, results


def main_oligo(argv: Sequence[str] | None = None) -> int:
    """Entry point for the classification launcher script."""
    parser = argparse.ArgumentParser(description="Train the oligo classifier.")
    _add_oligo_arguments(parser)
    args = parser.parse_args(argv)
    run_oligo(args)
    return 0


def main_interactions(argv: Sequence[str] | None = None) -> int:
    """Entry point for the interaction launcher script."""
    parser = argparse.ArgumentParser(
        description="Train the interaction regression models."
    )
    _add_interactions_arguments(parser)
    args = parser.parse_args(argv)
    args.dataset_roots = _resolve_interaction_roots(args)
    run_interactions(args)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the selected Deep-FISIK subcommand and return an exit code."""
    args = build_parser().parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        raise RuntimeError("No command handler was selected.")
    if args.command == "interactions":
        args.dataset_roots = _resolve_interaction_roots(args)
    handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
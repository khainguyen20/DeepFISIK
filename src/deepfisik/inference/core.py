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

"""Inference and evaluation entry points for Deep-FISIK.

This module provides command-line tools for running model predictions on
held-out data. The implementation reuses the same dataset transforms as the
training CLI so inference stays consistent with the trained models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import ConcatDataset, random_split
from torch_geometric.loader import DataLoader

from deepfisik.models.gnn_dimer import GNNDimer
from deepfisik.models.gnn_trimer import GNNTrimer
from deepfisik.models.gnn_tetramer import GNNTetramer
from deepfisik.models.gnn_oligo import GNNOligo

from deepfisik.cli import (
    DEFAULT_INTERACTION_ROOTS,
    DEFAULT_OLIGO_ROOTS,
    NormalizeCoordinates,
    transform_dimer,
    transform_monomer,
    transform_tetramer,
    transform_trimer,
)
from deepfisik.data.datasetInteractionsReadAll import SMI

OLIGO_TRANSFORMS = (
    transform_monomer,
    transform_dimer,
    transform_trimer,
    transform_tetramer,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_oligo_checkpoint() -> Path:
    return _repo_root() / "models" / "Images" / "classification" / "ImageClassificationModel.pt"


def _default_interactions_checkpoint(model: str) -> Path:
    checkpoints = {
        "dimer": _repo_root() / "models" / "Images" / "dimers" / "ImageDimerModel.pt",
        "trimer": _repo_root() / "models" / "PureSimulations" / "trimers" / "PureSimulationsTrimerModel.pt",
        "tetramer": _repo_root() / "models" / "PureSimulations" / "tetramers" / "PureSimulationsTetramerModel.pt",
    }
    try:
        return checkpoints[model]
    except KeyError as exc:
        raise ValueError(f"Unknown interaction model: {model}") from exc


def _device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRETRAINED_OLIGO_CHECKPOINTS = {
    (_repo_root() / "models" / "Images" / "classification" / "ImageClassificationModel.pt").resolve(),
    (_repo_root() / "models" / "PureSimulations" / "classification" / "PureSimulationsClassificationModel.pt").resolve()
}

def _is_pretrained_oligo_checkpoint(checkpoint: str | Path) -> bool:
    return Path(checkpoint).resolve() in PRETRAINED_OLIGO_CHECKPOINTS


def _pretrained_oligo_model_params(dataset) -> pd.DataFrame:
    # Fill these with the exact hyperparameters used for your pretrained
    # classification model.
    return pd.DataFrame(
        {
            "node_model_embedding_size": [72],
            "edge_model_embedding_size": [72],
            "model_attention_heads": [6],
            "model_attention_head_dimension": [72],
            "model_layers": [10],
            "model_attention_dropout_rate": [0],
            "model_top_k_ratio": [1.0],
            "model_top_k_every_n": [10],
            "model_edge_dim": dataset[0].edge_attr.shape[1],
            "model_feature_size": dataset[0].x.shape[1],
            "model_batch_size": 1,
            "model_layer_dropout_rate": [0],
            "model_laplacian": [10],
            "Epochs": 100,
            "E_Patience": 10,
            "Learning_Rate": 1e-4,
            "Weight_Decay": 1e-3,
            "Time_Step": 0.1,
            "Length_Dataset": len(dataset),
            "model_noClasses": 4,
        }
    )

def _pretrained_interaction_model_params(dataset) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "node_model_embedding_size": [96],
            "edge_model_embedding_size": [96],
            "model_attention_heads": [6],
            "model_attention_head_dimension": [96],
            "model_layers": [7],
            "model_attention_dropout_rate": [0],
            "model_top_k_ratio": [1.0],
            "model_top_k_every_n": [10],
            "model_edge_dim": dataset[0].edge_attr.shape[1],
            "model_feature_size": dataset[0].x.shape[1],
            "model_batch_size": 1,
            "model_layer_dropout_rate": [0],
            "model_laplacian": [10],
            "Epochs": 100,
            "E_Patience": 10,
            "Learning_Rate": 1e-4,
            "Weight_Decay": 1e-3,
            "Time_Step": 0.1,
            "Length_Dataset": len(dataset),
        }
    )

PRETRAINED_INTERACTION_CHECKPOINTS = {
    "dimer": {
        (_repo_root() / "models" / "Images" / "dimers" / "ImageDimerModel.pt").resolve(),
        (_repo_root() / "models" / "PureSimulations" / "dimers" / "PureSimulationsDimerModel.pt").resolve(),
    },
    "trimer": {
        (_repo_root() / "models" / "PureSimulations" / "trimers" / "PureSimulationsTrimerModel.pt").resolve(),
    },
    "tetramer": {
        (_repo_root() / "models" / "PureSimulations" / "tetramers" / "PureSimulationsTetramerModel.pt").resolve(),
    },
}

def _is_pretrained_interaction_checkpoint(checkpoint: str | Path, model_name: str) -> bool:
    return Path(checkpoint).resolve() in PRETRAINED_INTERACTION_CHECKPOINTS[model_name]

def _has_truth_fields(batch, field_names: Sequence[str]) -> bool:
    """Return True when a batch contains all requested target fields."""
    return all(hasattr(batch, field) for field in field_names)



def _load_state_dict_model(
    checkpoint: str | Path,
    device: torch.device,
    model: torch.nn.Module,
    model_name: str,
) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise TypeError(f"Expected a checkpoint dict or state_dict, got {type(ckpt)}")

    # Handle DataParallel / DistributedDataParallel checkpoints
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    # Backward-compatibility remap for renamed heads
    rename_maps = {
    "dimer": {
        "APOut": "AP2Out",
        "DROut": "DR2Out",
    },
    "trimer": {
        "AP2Out": "AP3Out",
        "AP1Out": "AP2Out",
        "DR2Out": "DR3Out",
        "DR1Out": "DR2Out",
    },
    "tetramer": {
        "AP3Out": "AP4Out",
        "AP2Out": "AP3Out",
        "AP1Out": "AP2Out",
        "DR3Out": "DR4Out",
        "DR2Out": "DR3Out",
        "DR1Out": "DR2Out",
    },
}

    rename_map = rename_maps.get(model_name, {})

    remapped_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        for old_name, new_name in rename_map.items():
            new_key = new_key.replace(old_name, new_name)
        remapped_state_dict[new_key] = v

    state_dict = remapped_state_dict

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def _load_model(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    model = torch.load(checkpoint, map_location=device)
    if hasattr(model, "module"):
        model = model.module
    model.to(device)
    model.eval()
    return model


def _split_dataset(dataset, split: str, seed: int):
    if split == "all":
        return dataset

    generator = torch.Generator().manual_seed(seed)
    train_len = int(np.ceil(0.8 * len(dataset)))
    test_len = len(dataset) - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len], generator=generator)
    return train_dataset if split == "train" else test_dataset


def _build_oligo_dataset(dataset_roots: Sequence[str], seed: int, split: str):
    datasets = [SMI(str(root), transform=transform) for root, transform in zip(dataset_roots, OLIGO_TRANSFORMS)]
    dataset = ConcatDataset(datasets)
    return _split_dataset(dataset, split, seed)


def _build_interactions_dataset(dataset_roots: Sequence[str], seed: int, split: str):
    datasets = [SMI(str(root), transform=NormalizeCoordinates) for root in dataset_roots]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return _split_dataset(dataset, split, seed)


def _flatten_cpu(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().reshape(-1).numpy()


def _safe_inverse_dc(dc: torch.Tensor, time_step: float) -> torch.Tensor:
    dc = torch.clamp(dc.float(), min=0.0)
    return ((dc / 30.0) ** 2) / (2.0 * time_step)


def _safe_inverse_linear(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.clamp(tensor.float() / scale, min=0.0)


def _classification_rows(batch, logits, label_offset: int) -> list[dict[str, float | int]]:
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    has_truth = hasattr(batch, "oligoLabel")
    true_labels = batch.oligoLabel.argmax(dim=-1) if has_truth else None

    rows: list[dict[str, float | int]] = []
    for idx in range(probs.size(0)):
        row: dict[str, float | int] = {
            "sample_index": label_offset + idx,
            "pred_class": int(preds[idx].item()),
        }
        if has_truth:
            row["true_class"] = int(true_labels[idx].item())
            row["correct"] = int(preds[idx].item() == true_labels[idx].item())
        for class_idx in range(probs.size(-1)):
            row[f"prob_class_{class_idx}"] = float(probs[idx, class_idx].item())
        rows.append(row)
    return rows


def _interaction_rows(
    batch,
    predictions: dict[str, torch.Tensor],
    truths: dict[str, torch.Tensor] | None,
    label_offset: int,
) -> list[dict[str, float | int]]:
    num_samples = next(iter(predictions.values())).shape[0]
    pred_arrays = {name: _flatten_cpu(tensor) for name, tensor in predictions.items()}
    truth_arrays = {name: _flatten_cpu(tensor) for name, tensor in truths.items()} if truths else {}

    rows: list[dict[str, float | int]] = []
    for idx in range(num_samples):
        row: dict[str, float | int] = {"sample_index": label_offset + idx}
        for name, array in truth_arrays.items():
            row[f"true_{name}"] = float(array[idx])
        for name, array in pred_arrays.items():
            row[f"pred_{name}"] = float(array[idx])
        rows.append(row)
    return rows


def _regression_metrics(predictions: pd.DataFrame, target_names: Sequence[str]) -> pd.DataFrame | None:
    metric_rows: list[dict[str, float | int | str]] = []
    if not len(predictions):
        return None

    for name in target_names:
        true_col = f"true_{name}"
        pred_col = f"pred_{name}"
        if true_col not in predictions.columns or pred_col not in predictions.columns:
            continue

        y_true = predictions[true_col]
        y_pred = predictions[pred_col]
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mean_true = float(np.mean(y_true))
        metric_rows.append(
            {
                "target": name,
                "r2": float(r2_score(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "normalized_rmse": float(rmse / mean_true) if mean_true != 0 else float("nan"),
                "num_samples": int(len(predictions)),
            }
        )

    return pd.DataFrame(metric_rows) if metric_rows else None


def run_oligo_inference(args: argparse.Namespace):
    """Run inference for the oligo classification model and save predictions.

    If the batch contains ground-truth labels, accuracy is computed and a metrics
    file is saved. Otherwise, only predictions are written.
    """
    device = _device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = _build_oligo_dataset(args.dataset_roots, args.seed, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    checkpoint = Path(args.checkpoint) if args.checkpoint else _default_oligo_checkpoint()

    if _is_pretrained_oligo_checkpoint(checkpoint):
        model_params = _pretrained_oligo_model_params(dataset)
        model = GNNOligo(model_params=model_params)
        model = _load_state_dict_model(checkpoint, device, model,"classification")
    else:
        model = _load_model(checkpoint, device)
    rows: list[dict[str, float | int]] = []
    with torch.no_grad():
        offset = 0
        for batch in loader:
            outputs = model(
                batch.x.to(device),
                batch.edge_index.long().to(device),
                batch.edge_attr.to(device),
                batch.batch.to(device),
                device,
            )
            logits = outputs[2]
            rows.extend(_classification_rows(batch, logits, offset))
            offset += batch.num_graphs

    predictions = pd.DataFrame(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / f"predictions_{args.date}.csv", index=False)

    accuracy = None
    if len(predictions) and "true_class" in predictions.columns:
        predictions["correct"] = predictions["correct"].astype(int)
        accuracy = accuracy_score(predictions["true_class"], predictions["pred_class"])
        pd.DataFrame([{"accuracy": accuracy, "num_samples": len(predictions)}]).to_csv(
            output_dir / f"metrics_{args.date}.csv",
            index=False,
        )

    return predictions, accuracy


def _run_interactions_batch_dimer(model, batch, device, time_step: float):
    node_out, edge_out, global_out, dc, ap2, dr2, rd, lf = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        device,
    )
    predictions = {
        "DC": _safe_inverse_dc(dc, time_step),
        "AP2": _safe_inverse_linear(ap2, 50.0),
        "DR2": _safe_inverse_linear(dr2, 20.0 * time_step),
        "RD": torch.clamp(rd.float(), min=0.0),
        "LF": _safe_inverse_linear(lf, 10.0),
    }
    truths = (
        {
            "DC": batch.DC.float(),
            "AP2": batch.AP2.float(),
            "DR2": batch.DR2.float(),
            "RD": batch.RD.float(),
            "LF": batch.LF.float(),
        }
        if _has_truth_fields(batch, ("DC", "AP2", "DR2", "RD", "LF"))
        else None
    )
    return predictions, truths


def _run_interactions_batch_trimer(model, batch, device, time_step: float):
    node_out, edge_out, global_out, dc, ap2, ap3, dr2, dr3, rd, lf = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        device,
    )
    predictions = {
        "DC": _safe_inverse_dc(dc, time_step),
        "AP2": _safe_inverse_linear(ap2, 50.0),
        "AP3": _safe_inverse_linear(ap3, 50.0),
        "DR2": _safe_inverse_linear(dr2, 20.0 * time_step),
        "DR3": _safe_inverse_linear(dr3, 20.0 * time_step),
        "RD": torch.clamp(rd.float(), min=0.0),
        "LF": _safe_inverse_linear(lf, 10.0),
    }
    truths = (
        {
            "DC": batch.DC.float(),
            "AP2": batch.AP2.float(),
            "AP3": batch.AP3.float(),
            "DR2": batch.DR2.float(),
            "DR3": batch.DR3.float(),
            "RD": batch.RD.float(),
            "LF": batch.LF.float(),
        }
        if _has_truth_fields(batch, ("DC", "AP2","AP3" ,"DR2","DR3", "RD", "LF"))
        else None
    )
    return predictions, truths


def _run_interactions_batch_tetramer(model, batch, device, time_step: float):
    node_out, edge_out, global_out, dc, ap2, ap3, ap4, dr2, dr3, dr4, rd, lf = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        device,
    )
    predictions = {
        "DC": _safe_inverse_dc(dc, time_step),
        "AP2": _safe_inverse_linear(ap2, 50.0),
        "AP3": _safe_inverse_linear(ap3, 50.0),
        "AP4": _safe_inverse_linear(ap4, 50.0),
        "DR2": _safe_inverse_linear(dr2, 20.0 * time_step),
        "DR3": _safe_inverse_linear(dr3, 20.0 * time_step),
        "DR4": _safe_inverse_linear(dr4, 20.0 * time_step),
        "RD": torch.clamp(rd.float(), min=0.0),
        "LF": _safe_inverse_linear(lf, 10.0),
    }
    truths = (
        {
            "DC": batch.DC.float(),
            "AP2": batch.AP2.float(),
            "AP3": batch.AP3.float(),
            "AP4": batch.AP4.float(),
            "DR2": batch.DR2.float(),
            "DR3": batch.DR3.float(),
            "DR4": batch.DR4.float(),
            "RD": batch.RD.float(),
            "LF": batch.LF.float(),
        }
        if _has_truth_fields(batch, ("DC", "AP2", "AP3", "AP4", "DR2", "DR3", "DR4", "RD", "LF"))
        else None
    )
    return predictions, truths


def _run_interactions_inference_generic(
    args: argparse.Namespace, model_name: str, batch_runner, target_names: Sequence[str]
):
    device = _device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = _build_interactions_dataset(args.dataset_roots, args.seed, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    checkpoint = Path(args.checkpoint) if args.checkpoint else _default_interactions_checkpoint(model_name)

    model_registry = {
        "dimer": GNNDimer,
        "trimer": GNNTrimer,
        "tetramer": GNNTetramer,
    }

    if _is_pretrained_interaction_checkpoint(checkpoint, model_name):
        model_params = _pretrained_interaction_model_params(dataset)
        model = model_registry[model_name](model_params=model_params)
        model = _load_state_dict_model(checkpoint, device, model, model_name)
    else:
        model = _load_model(checkpoint, device)

    rows: list[dict[str, float | int]] = []
    with torch.no_grad():
        offset = 0
        for batch in loader:
            predictions, truths = batch_runner(model, batch, device, args.time_step)
            rows.extend(_interaction_rows(batch, predictions, truths, offset))
            offset += batch.num_graphs

    predictions_df = pd.DataFrame(rows)
    metrics_df = _regression_metrics(predictions_df, target_names)

    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_dir / f"predictions_{args.date}.csv", index=False)
    if metrics_df is not None:
        metrics_df.to_csv(output_dir / f"metrics_{args.date}.csv", index=False)
    return predictions_df, metrics_df


def run_dimer_inference(args: argparse.Namespace):
    """Run inference for the dimer regression model and save predictions."""
    return _run_interactions_inference_generic(
        args,
        "dimer",
        _run_interactions_batch_dimer,
        ("DC", "AP2", "DR2", "RD", "LF"),
    )


def run_trimer_inference(args: argparse.Namespace):
    """Run inference for the trimer regression model and save predictions."""
    return _run_interactions_inference_generic(
        args,
        "trimer",
        _run_interactions_batch_trimer,
        ("DC", "AP2", "AP3", "DR2", "DR3", "RD", "LF"),
    )


def run_tetramer_inference(args: argparse.Namespace):
    """Run inference for the tetramer regression model and save predictions."""
    return _run_interactions_inference_generic(
        args,
        "tetramer",
        _run_interactions_batch_tetramer,
        ("DC", "AP2", "AP3", "AP4", "DR2", "DR3", "DR4", "RD", "LF"),
    )


def run_interactions_inference(args: argparse.Namespace):
    """Run inference for the selected interaction regression model."""
    runners = {
        "dimer": run_dimer_inference,
        "trimer": run_trimer_inference,
        "tetramer": run_tetramer_inference,
    }
    try:
        runner = runners[args.model]
    except KeyError as exc:
        raise ValueError(f"Unknown interaction model: {args.model}") from exc
    return runner(args)


def build_oligo_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for oligo inference."""
    parser = argparse.ArgumentParser(description="Run inference for the oligo classifier. Labels are optional; if present, accuracy is saved.")
    parser.add_argument("--checkpoint", default=None, help="Path to a saved classifier model. Defaults to the bundled checkpoint.")
    parser.add_argument(
        "--dataset-roots",
        nargs=4,
        metavar=("MONO", "DIMER", "TRIMER", "TETRAMER"),
        default=list(DEFAULT_OLIGO_ROOTS),
        help="Paths to the monomer, dimer, trimer, and tetramer datasets.",
    )
    parser.add_argument("--output-dir", default="runs/inference/oligo")
    parser.add_argument("--date", default="run")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    return parser


def build_interactions_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for interaction inference."""
    parser = argparse.ArgumentParser(description="Run inference for a dimer, trimer, or tetramer regression model. If labels are present, R2/MAE/normalized RMSE are saved; otherwise only predictions are written.")
    parser.add_argument("--checkpoint", default=None, help="Path to a saved interaction model. Defaults to the bundled checkpoint for the selected model.")
    parser.add_argument(
        "--model",
        choices=("dimer", "trimer", "tetramer"),
        default="tetramer",
        help="Name of the interaction model to evaluate.",
    )
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        default=[DEFAULT_INTERACTION_ROOTS],
        metavar="ROOT",
        help="One or more paths to interaction datasets. All roots are concatenated.",
    )
    parser.add_argument("--output-dir", default="runs/inference/interactions")
    parser.add_argument("--date", default="run")
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    return parser


def main_oligo(argv: Sequence[str] | None = None) -> int:
    """Entry point for the oligo inference launcher."""
    args = build_oligo_parser().parse_args(argv)
    run_oligo_inference(args)
    return 0


def main_interactions(argv: Sequence[str] | None = None) -> int:
    """Entry point for the interaction inference launcher."""
    args = build_interactions_parser().parse_args(argv)
    run_interactions_inference(args)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch inference subcommands from the command line."""
    parser = argparse.ArgumentParser(description="Deep-FISIK inference command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    oligo = subparsers.add_parser("classification", help="Run oligo classification inference.")
    for action in build_oligo_parser()._actions[1:]:
        oligo._add_action(action)
    oligo.set_defaults(handler=run_oligo_inference)

    interactions = subparsers.add_parser("interactions", help="Run interaction inference.")
    for action in build_interactions_parser()._actions[1:]:
        interactions._add_action(action)
    interactions.set_defaults(handler=run_interactions_inference)

    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        raise RuntimeError("No inference handler was selected.")
    handler(args)
    return 0

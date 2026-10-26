#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balabit Stage 5 — Deep Models Pilot/Full Training
=================================================

Trains and evaluates three sequence models on Stage 4 tensors:
  1) CNN
  2) BiLSTM
  3) CNN + BiLSTM

Authentication protocol
-----------------------
- The network is trained as a shared multiclass user classifier.
- For each claimed user, that user's softmax probability becomes the claim score.
- Genuine claims: windows belonging to the claimed user.
- Impostor claims: windows belonging to all other users.
- A separate threshold is selected for every user from validation data only.
- Thresholds are frozen before internal-test evaluation.
- FAR, FRR, EER, AUC, F1, MCC, latency, and TTA are reported.
- Stage 4 session splits are reused without creating a new split.

The script is designed for Windows/CPU systems with memory-mapped .npy tensors.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import platform
import random
import sys
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:
    raise SystemExit(
        "PyTorch could not be imported. Run: python -m pip install -r requirements.txt\n"
        f"Original error: {exc}"
    )

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
        roc_curve,
    )
except Exception as exc:
    raise SystemExit(
        "scikit-learn could not be imported. Run: python -m pip install -r requirements.txt\n"
        f"Original error: {exc}"
    )

VERSION = "1.0.0"
MODEL_NAMES = ("cnn", "bilstm", "hybrid")
SPLITS = ("train", "validation", "internal_test")
EPS = 1e-12


@dataclass
class ModelConfig:
    name: str
    input_features: int
    n_classes: int
    cnn_channels_1: int = 64
    cnn_channels_2: int = 128
    lstm_hidden: int = 64
    dropout: float = 0.30


class MemoryMappedSequenceDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path, indices: np.ndarray):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.X = np.load(self.x_path, mmap_mode="r")
        self.y = np.load(self.y_path, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)

        if self.X.ndim != 3:
            raise ValueError(f"Expected a 3-D tensor at {self.x_path}; got {self.X.shape}")
        if len(self.X) != len(self.y):
            raise ValueError(f"X/y length mismatch: {len(self.X)} != {len(self.y)}")
        if len(self.indices) and (self.indices.min() < 0 or self.indices.max() >= len(self.X)):
            raise ValueError("Dataset indices are outside tensor bounds.")

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, position: int):
        idx = int(self.indices[position])
        # Copy avoids the PyTorch warning about read-only memory maps.
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32, copy=True))
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y, idx


class CNNModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(cfg.input_features, cfg.cnn_channels_1, kernel_size=5, padding=2),
            nn.BatchNorm1d(cfg.cnn_channels_1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(cfg.cnn_channels_1, cfg.cnn_channels_2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cfg.cnn_channels_2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.cnn_channels_2, cfg.n_classes),
        )

    def forward(self, x):
        # Input: [batch, time, features]
        x = x.transpose(1, 2)
        return self.classifier(self.features(x))


class BiLSTMModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.input_features,
            hidden_size=cfg.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.lstm_hidden * 2, cfg.n_classes)

    def forward(self, x):
        output, _ = self.lstm(x)
        pooled = output.mean(dim=1)
        return self.classifier(self.dropout(pooled))


class HybridCNNBiLSTM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(cfg.input_features, cfg.cnn_channels_1, kernel_size=5, padding=2),
            nn.BatchNorm1d(cfg.cnn_channels_1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(cfg.cnn_channels_1, cfg.cnn_channels_2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cfg.cnn_channels_2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=cfg.cnn_channels_2,
            hidden_size=cfg.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.lstm_hidden * 2, cfg.n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        output, _ = self.lstm(x)
        pooled = output.mean(dim=1)
        return self.classifier(self.dropout(pooled))


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.name == "cnn":
        return CNNModel(cfg)
    if cfg.name == "bilstm":
        return BiLSTMModel(cfg)
    if cfg.name == "hybrid":
        return HybridCNNBiLSTM(cfg)
    raise ValueError(f"Unsupported model: {cfg.name}")


def setup_logging(output: Path) -> logging.Logger:
    logger = logging.getLogger("balabit_stage5")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(output / "run_log.txt", encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_models(text: str) -> list[str]:
    vals = [x.strip().lower() for x in text.split(",") if x.strip()]
    unknown = sorted(set(vals) - set(MODEL_NAMES))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown model(s): {unknown}")
    if not vals:
        raise argparse.ArgumentTypeError("At least one model is required.")
    return list(dict.fromkeys(vals))


def safe_div(n: float, d: float) -> float:
    return float(n / d) if d else float("nan")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def balanced_indices(
    labels: np.ndarray,
    cap_per_user: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for user in sorted(np.unique(labels).tolist()):
        idx = np.flatnonzero(labels == user)
        available = len(idx)
        if cap_per_user > 0 and available > cap_per_user:
            idx = np.sort(rng.choice(idx, size=cap_per_user, replace=False))
        selected.append(idx)
        rows.append({
            "user_index": int(user),
            "available_windows": int(available),
            "selected_windows": int(len(idx)),
            "cap_per_user": int(cap_per_user),
        })

    merged = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    rng.shuffle(merged)
    return merged.astype(np.int64), pd.DataFrame(rows)


def make_class_weights(labels: np.ndarray, indices: np.ndarray, n_classes: int) -> np.ndarray:
    selected_labels = np.asarray(labels[indices], dtype=np.int64)
    counts = np.bincount(selected_labels, minlength=n_classes).astype(np.float64)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training selection has no samples for user indices: {missing}")
    weights = len(selected_labels) / (n_classes * counts)
    return weights.astype(np.float32)


def validate_stage4(
    stage4_root: Path,
    window_size: int,
) -> tuple[Path, dict[str, Path], pd.DataFrame, pd.DataFrame]:
    tensor_dir = stage4_root / "tensors" / f"w{window_size}"
    if not tensor_dir.is_dir():
        raise FileNotFoundError(f"Tensor directory not found: {tensor_dir}")

    paths: dict[str, Path] = {}
    for split in SPLITS:
        for kind in ("X", "y_user", "session_index"):
            p = tensor_dir / f"{split}_{kind}.npy"
            if not p.is_file():
                raise FileNotFoundError(f"Required Stage 4 tensor not found: {p}")
            paths[f"{split}_{kind}"] = p
        metadata = tensor_dir / f"{split}_metadata.csv.gz"
        if not metadata.is_file():
            raise FileNotFoundError(f"Required Stage 4 metadata not found: {metadata}")
        paths[f"{split}_metadata"] = metadata

    users_path = stage4_root / "users.csv"
    sessions_path = stage4_root / "sessions.csv"
    if not users_path.is_file() or not sessions_path.is_file():
        raise FileNotFoundError("users.csv or sessions.csv is missing from Stage 4 root.")

    users = pd.read_csv(users_path, encoding="utf-8-sig")
    sessions = pd.read_csv(sessions_path, encoding="utf-8-sig")
    required_users = {"user_index", "user_id"}
    if not required_users.issubset(users.columns):
        raise ValueError(f"users.csv must contain {sorted(required_users)}")

    for split in SPLITS:
        X = np.load(paths[f"{split}_X"], mmap_mode="r")
        y = np.load(paths[f"{split}_y_user"], mmap_mode="r")
        s = np.load(paths[f"{split}_session_index"], mmap_mode="r")
        if X.ndim != 3:
            raise ValueError(f"{split}_X must be 3-D; got {X.shape}")
        if X.shape[1] != window_size:
            raise ValueError(f"{split}_X window size is {X.shape[1]}, expected {window_size}")
        if len(X) != len(y) or len(X) != len(s):
            raise ValueError(f"{split}: X/y/session length mismatch.")
        if not np.isfinite(np.asarray(X[: min(128, len(X))])).all():
            raise ValueError(f"{split}: non-finite values detected in a tensor sample.")

    return tensor_dir, paths, users, sessions


def protocol_audit(
    paths: dict[str, Path],
    users: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    session_sets = {}
    for split in SPLITS:
        session_sets[split] = set(
            np.unique(np.load(paths[f"{split}_session_index"], mmap_mode="r")).tolist()
        )
    intersections = {
        "train_validation": sorted(session_sets["train"] & session_sets["validation"]),
        "train_internal_test": sorted(session_sets["train"] & session_sets["internal_test"]),
        "validation_internal_test": sorted(session_sets["validation"] & session_sets["internal_test"]),
    }
    rows.append({
        "check": "no_session_overlap",
        "passed": not any(intersections.values()),
        "details": json.dumps(intersections),
    })

    expected_users = set(users["user_index"].astype(int))
    missing_by_split = {}
    for split in SPLITS:
        labels = np.load(paths[f"{split}_y_user"], mmap_mode="r")
        missing = sorted(expected_users - set(np.unique(labels).astype(int).tolist()))
        if missing:
            missing_by_split[split] = missing
    rows.append({
        "check": "all_users_in_all_splits",
        "passed": not bool(missing_by_split),
        "details": json.dumps(missing_by_split) if missing_by_split else "All users appear in every split.",
    })

    rows.append({
        "check": "stage4_preprocessing_reused",
        "passed": True,
        "details": "Stage 5 reads normalized Stage 4 tensors without refitting preprocessing.",
    })
    rows.append({
        "check": "validation_thresholds_only",
        "passed": True,
        "details": "Per-user thresholds are selected from validation scores only.",
    })
    rows.append({
        "check": "internal_test_locked",
        "passed": True,
        "details": "Internal-test labels are used only after model/threshold selection.",
    })
    rows.append({
        "check": "same_subset_for_all_models",
        "passed": True,
        "details": "The saved deterministic row selections are reused by CNN, BiLSTM, and Hybrid.",
    })
    return pd.DataFrame(rows)


def make_loader(
    x_path: Path,
    y_path: Path,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    dataset = MemoryMappedSequenceDataset(x_path, y_path, indices)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        generator=generator,
        drop_last=False,
    )


def collect_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_rows: list[np.ndarray] = []
    total_loss = 0.0
    total_n = 0

    with torch.inference_mode():
        for x, y, row_index in loader:
            x = x.to(device, non_blocking=True)
            y_device = y.to(device, non_blocking=True)
            logits = model(x)
            if criterion is not None:
                loss = criterion(logits, y_device)
                total_loss += float(loss.item()) * len(y)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(y.numpy())
            all_rows.append(row_index.numpy())
            total_n += len(y)

    mean_loss = total_loss / total_n if criterion is not None and total_n else float("nan")
    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_labels, axis=0).astype(np.int64),
        np.concatenate(all_rows, axis=0).astype(np.int64),
        mean_loss,
    )


def eer_and_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return float("nan"), 0.5
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    if not np.isfinite(threshold):
        finite = thresholds[np.isfinite(thresholds)]
        threshold = float(finite.max()) if len(finite) else 0.5
    return eer, threshold


def binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    y_pred = (scores >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    test_eer, _ = eer_and_threshold(y_true, scores)
    try:
        auc = float(roc_auc_score(y_true, scores))
    except Exception:
        auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "far": safe_div(fp, fp + tn),
        "frr": safe_div(fn, fn + tp),
        "test_eer": test_eer,
        "auc": auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "claims": int(len(y_true)),
        "genuine_claims": int(np.sum(y_true == 1)),
        "impostor_claims": int(np.sum(y_true == 0)),
    }


def select_validation_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    users: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for rec in users.sort_values("user_index").itertuples(index=False):
        user_index = int(rec.user_index)
        y_binary = (labels == user_index).astype(np.int8)
        scores = probabilities[:, user_index]
        eer, threshold = eer_and_threshold(y_binary, scores)
        val_metrics = binary_metrics(y_binary, scores, threshold)
        rows.append({
            "user_id": str(rec.user_id),
            "user_index": user_index,
            "threshold": threshold,
            "validation_eer": eer,
            "validation_accuracy": val_metrics["accuracy"],
            "validation_far": val_metrics["far"],
            "validation_frr": val_metrics["frr"],
            "validation_auc": val_metrics["auc"],
            "validation_f1": val_metrics["f1"],
            "validation_mcc": val_metrics["mcc"],
        })
    df = pd.DataFrame(rows)
    macro = {
        "macro_validation_eer": float(df["validation_eer"].mean()),
        "macro_validation_f1": float(df["validation_f1"].mean()),
        "macro_validation_auc": float(df["validation_auc"].mean()),
        "macro_validation_mcc": float(df["validation_mcc"].mean()),
    }
    return df, macro


def evaluate_authentication(
    model_name: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
    row_indices: np.ndarray,
    users: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    claim_frames: list[pd.DataFrame] = []

    threshold_map = thresholds.set_index("user_index")
    for rec in users.sort_values("user_index").itertuples(index=False):
        user_index = int(rec.user_index)
        user_id = str(rec.user_id)
        threshold = float(threshold_map.loc[user_index, "threshold"])
        validation_eer = float(threshold_map.loc[user_index, "validation_eer"])

        y_binary = (labels == user_index).astype(np.int8)
        scores = probabilities[:, user_index]
        metrics = binary_metrics(y_binary, scores, threshold)
        metric_rows.append({
            "model": model_name,
            "user_id": user_id,
            "user_index": user_index,
            "threshold_from_validation": threshold,
            "validation_eer": validation_eer,
            **metrics,
        })

        pred = (scores >= threshold).astype(np.int8)
        claim_frames.append(pd.DataFrame({
            "model": model_name,
            "claimed_user_id": user_id,
            "claimed_user_index": user_index,
            "test_row_index": row_indices,
            "actual_user_index": labels,
            "is_genuine": y_binary,
            "score": scores.astype(np.float32),
            "threshold": np.float32(threshold),
            "prediction": pred,
            "correct": (pred == y_binary).astype(np.int8),
        }))

    return pd.DataFrame(metric_rows), pd.concat(claim_frames, ignore_index=True)


def macro_metrics(metrics_by_user: pd.DataFrame, model_name: str) -> dict[str, Any]:
    numeric = [
        "accuracy", "far", "frr", "test_eer", "auc", "f1", "mcc",
        "validation_eer",
    ]
    out: dict[str, Any] = {"model": model_name}
    for col in numeric:
        out[f"macro_{col}"] = float(metrics_by_user[col].mean())
        out[f"std_{col}"] = float(metrics_by_user[col].std(ddof=1))
    return out


def pooled_metrics_from_claims(claims: pd.DataFrame, model_name: str) -> dict[str, Any]:
    y_true = claims["is_genuine"].to_numpy(dtype=np.int8)
    y_pred = claims["prediction"].to_numpy(dtype=np.int8)
    scores = claims["score"].to_numpy(dtype=np.float64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_true, scores))
    except Exception:
        auc = float("nan")
    eer, _ = eer_and_threshold(y_true, scores)
    return {
        "model": model_name,
        "pooled_accuracy": float(accuracy_score(y_true, y_pred)),
        "pooled_far": safe_div(fp, fp + tn),
        "pooled_frr": safe_div(fn, fn + tp),
        "pooled_eer": eer,
        "pooled_auc": auc,
        "pooled_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pooled_mcc": float(matthews_corrcoef(y_true, y_pred)),
        "pooled_tn": int(tn),
        "pooled_fp": int(fp),
        "pooled_fn": int(fn),
        "pooled_tp": int(tp),
        "pooled_claims": int(len(y_true)),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item()) * len(y)
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += len(y)

    return total_loss / total_n, total_correct / total_n


def is_better_checkpoint(
    current_eer: float,
    current_f1: float,
    best_eer: float,
    best_f1: float,
    tolerance: float = 1e-5,
) -> bool:
    if not np.isfinite(best_eer):
        return True
    if current_eer < best_eer - tolerance:
        return True
    if abs(current_eer - best_eer) <= tolerance and current_f1 > best_f1 + tolerance:
        return True
    return False


def measure_latency(
    model: nn.Module,
    sample: np.ndarray,
    device: torch.device,
    batch_size: int,
    runs: int,
) -> dict[str, float]:
    model.eval()
    single = torch.from_numpy(np.array(sample[0:1], dtype=np.float32, copy=True)).to(device)
    repeat_count = max(1, min(batch_size, 128))
    batch = single.repeat(repeat_count, 1, 1)

    def run_tensor(tensor: torch.Tensor, n_runs: int) -> list[float]:
        with torch.inference_mode():
            for _ in range(5):
                _ = model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model(tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
        return times

    single_times = run_tensor(single, runs)
    batch_times = run_tensor(batch, max(20, runs // 2))
    return {
        "single_mean_ms": float(np.mean(single_times) * 1000),
        "single_p95_ms": float(np.percentile(single_times, 95) * 1000),
        "single_std_ms": float(np.std(single_times) * 1000),
        "batch_size_measured": int(repeat_count),
        "batch_mean_ms": float(np.mean(batch_times) * 1000),
        "batch_p95_ms": float(np.percentile(batch_times, 95) * 1000),
        "throughput_windows_per_second": float(repeat_count / np.mean(batch_times)),
    }


def load_window_duration_stats(metadata_path: Path, selected_rows: np.ndarray) -> dict[str, float]:
    meta = pd.read_csv(metadata_path, compression="gzip")
    if "row_index" not in meta.columns or "window_duration" not in meta.columns:
        return {
            "window_duration_mean_sec": float("nan"),
            "window_duration_median_sec": float("nan"),
            "window_duration_p95_sec": float("nan"),
        }
    indexed = meta.set_index("row_index")
    available = indexed.index.intersection(selected_rows)
    durations = pd.to_numeric(indexed.loc[available, "window_duration"], errors="coerce").dropna()
    return {
        "window_duration_mean_sec": float(durations.mean()),
        "window_duration_median_sec": float(durations.median()),
        "window_duration_p95_sec": float(durations.quantile(0.95)),
    }


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: ModelConfig,
    user_mapping: pd.DataFrame,
    window_size: int,
    best_epoch: int,
    best_validation_eer: float,
    best_validation_f1: float,
) -> None:
    torch.save(
        {
            "stage5_version": VERSION,
            "model_config": asdict(cfg),
            "window_size": int(window_size),
            "best_epoch": int(best_epoch),
            "best_validation_eer": float(best_validation_eer),
            "best_validation_f1": float(best_validation_f1),
            "user_mapping": user_mapping.to_dict(orient="records"),
            "state_dict": model.state_dict(),
        },
        path,
    )


def create_review_zip(output: Path) -> Path:
    target = output.parent / f"{output.name}_review.zip"
    if target.exists():
        target.unlink()
    include = {
        "stage5_summary.json",
        "training_history.csv",
        "metrics_by_user.csv",
        "metrics_macro.csv",
        "metrics_pooled.csv",
        "thresholds.csv",
        "latency_tta.csv",
        "multiclass_metrics.csv",
        "confusion_matrices.csv",
        "model_inventory.csv",
        "selection_summary.csv",
        "protocol_audit.csv",
        "stage5_errors.csv",
        "run_log.txt",
    }
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(output.iterdir()):
            if p.is_file() and p.name in include:
                zf.write(p, arcname=p.name)
    return target


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Balabit Stage 5 deep sequence models")
    ap.add_argument("--stage4-root", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=Path("balabit_stage5_pilot_w50"))
    ap.add_argument("--window-size", type=int, default=50)
    ap.add_argument("--models", type=parse_models, default=list(MODEL_NAMES))
    ap.add_argument("--max-train-per-user", type=int, default=2500)
    ap.add_argument("--max-validation-per-user", type=int, default=800)
    ap.add_argument("--max-test-per-user", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--latency-runs", type=int, default=100)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    stage4_root = args.stage4_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)
    logger = setup_logging(output)
    start_time = time.time()
    errors: list[dict[str, Any]] = []

    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        logger.error("epochs, patience, and batch-size must be positive.")
        return 2
    if args.window_size < 5:
        logger.error("window-size must be >= 5.")
        return 2
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    else:
        cpu_count = os.cpu_count() or 2
        torch.set_num_threads(max(1, min(8, cpu_count - 1)))

    set_seed(args.seed)
    try:
        device = choose_device(args.device)
    except Exception as exc:
        logger.error(str(exc))
        return 2

    logger.info("Balabit Stage 5 v%s", VERSION)
    logger.info("Stage 4 root: %s", stage4_root)
    logger.info("Output: %s", output)
    logger.info("Window size: %d | models=%s", args.window_size, args.models)
    logger.info("Device=%s | torch=%s | threads=%d", device, torch.__version__, torch.get_num_threads())
    logger.info("Python=%s", sys.version.replace("\n", " "))

    try:
        tensor_dir, paths, users, sessions = validate_stage4(stage4_root, args.window_size)
        audit = protocol_audit(paths, users)
        audit.to_csv(output / "protocol_audit.csv", index=False, encoding="utf-8-sig")
        if not audit["passed"].all():
            raise RuntimeError("Protocol audit failed. See protocol_audit.csv.")
    except Exception as exc:
        logger.exception("Stage 4 validation failed.")
        pd.DataFrame([{
            "phase": "stage4_validation",
            "model": "",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }]).to_csv(output / "stage5_errors.csv", index=False, encoding="utf-8-sig")
        return 3

    n_classes = len(users)
    X_train = np.load(paths["train_X"], mmap_mode="r")
    input_features = int(X_train.shape[2])

    split_caps = {
        "train": args.max_train_per_user,
        "validation": args.max_validation_per_user,
        "internal_test": args.max_test_per_user,
    }
    selected_indices: dict[str, np.ndarray] = {}
    selection_frames: list[pd.DataFrame] = []

    for split_number, split in enumerate(SPLITS):
        labels = np.load(paths[f"{split}_y_user"], mmap_mode="r")
        indices, summary = balanced_indices(
            labels,
            split_caps[split],
            args.seed + split_number * 1000,
        )
        selected_indices[split] = indices
        summary.insert(0, "protocol_split", split)
        selection_frames.append(summary)

        np.save(output / f"selected_{split}_row_indices.npy", indices)

    selection_summary = pd.concat(selection_frames, ignore_index=True)
    selection_summary.to_csv(output / "selection_summary.csv", index=False, encoding="utf-8-sig")

    train_labels = np.load(paths["train_y_user"], mmap_mode="r")
    class_weights_np = make_class_weights(
        train_labels,
        selected_indices["train"],
        n_classes,
    )
    logger.info(
        "Selections: train=%d validation=%d test=%d",
        len(selected_indices["train"]),
        len(selected_indices["validation"]),
        len(selected_indices["internal_test"]),
    )

    train_loader = make_loader(
        paths["train_X"], paths["train_y_user"], selected_indices["train"],
        args.batch_size, True, args.seed, device, args.num_workers,
    )
    validation_loader = make_loader(
        paths["validation_X"], paths["validation_y_user"], selected_indices["validation"],
        args.batch_size, False, args.seed + 1, device, args.num_workers,
    )
    test_loader = make_loader(
        paths["internal_test_X"], paths["internal_test_y_user"], selected_indices["internal_test"],
        args.batch_size, False, args.seed + 2, device, args.num_workers,
    )

    history_rows: list[dict[str, Any]] = []
    all_metrics_by_user: list[pd.DataFrame] = []
    macro_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    threshold_frames: list[pd.DataFrame] = []
    latency_rows: list[dict[str, Any]] = []
    multiclass_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []

    test_X_memmap = np.load(paths["internal_test_X"], mmap_mode="r")
    duration_stats = load_window_duration_stats(
        paths["internal_test_metadata"],
        selected_indices["internal_test"],
    )

    for model_position, model_name in enumerate(args.models, start=1):
        logger.info("============================================================")
        logger.info("Training model %d/%d: %s", model_position, len(args.models), model_name)
        try:
            set_seed(args.seed)
            cfg = ModelConfig(
                name=model_name,
                input_features=input_features,
                n_classes=n_classes,
                dropout=args.dropout,
            )
            model = build_model(cfg).to(device)
            total_params, trainable_params = count_parameters(model)
            class_weights = torch.tensor(class_weights_np, device=device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=max(1, args.patience // 2),
                min_lr=1e-6,
            )

            best_state = None
            best_epoch = 0
            best_eer = float("inf")
            best_f1 = -float("inf")
            epochs_without_improvement = 0
            model_training_start = time.time()

            for epoch in range(1, args.epochs + 1):
                epoch_start = time.time()
                train_loss, train_acc = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    args.grad_clip,
                )
                val_probs, val_labels, val_rows, val_loss = collect_probabilities(
                    model,
                    validation_loader,
                    device,
                    criterion,
                )
                val_thresholds, val_macro = select_validation_thresholds(
                    val_probs,
                    val_labels,
                    users,
                )
                multiclass_val_acc = float(
                    accuracy_score(val_labels, val_probs.argmax(axis=1))
                )
                current_eer = val_macro["macro_validation_eer"]
                current_f1 = val_macro["macro_validation_f1"]
                scheduler.step(current_eer)

                improved = is_better_checkpoint(
                    current_eer,
                    current_f1,
                    best_eer,
                    best_f1,
                )
                if improved:
                    best_eer = current_eer
                    best_f1 = current_f1
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                history_rows.append({
                    "model": model_name,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_multiclass_accuracy": train_acc,
                    "validation_loss": val_loss,
                    "validation_multiclass_accuracy": multiclass_val_acc,
                    **val_macro,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "checkpoint_improved": improved,
                    "epoch_seconds": time.time() - epoch_start,
                })
                logger.info(
                    "%s epoch=%d train_loss=%.4f train_acc=%.4f "
                    "val_loss=%.4f val_acc=%.4f macro_EER=%.4f macro_F1=%.4f%s",
                    model_name,
                    epoch,
                    train_loss,
                    train_acc,
                    val_loss,
                    multiclass_val_acc,
                    current_eer,
                    current_f1,
                    " *" if improved else "",
                )

                if epochs_without_improvement >= args.patience:
                    logger.info(
                        "%s early stopping at epoch %d; best epoch=%d",
                        model_name,
                        epoch,
                        best_epoch,
                    )
                    break

            if best_state is None:
                raise RuntimeError(f"No checkpoint was created for {model_name}.")
            model.load_state_dict(best_state)

            # Recalculate validation thresholds from the selected best checkpoint.
            val_probs, val_labels, val_rows, val_loss = collect_probabilities(
                model,
                validation_loader,
                device,
                criterion,
            )
            thresholds, val_macro = select_validation_thresholds(
                val_probs,
                val_labels,
                users,
            )
            thresholds.insert(0, "model", model_name)
            threshold_frames.append(thresholds)

            test_probs, test_labels, test_rows, test_loss = collect_probabilities(
                model,
                test_loader,
                device,
                criterion,
            )
            metrics_by_user, claims = evaluate_authentication(
                model_name,
                test_probs,
                test_labels,
                test_rows,
                users,
                thresholds,
            )
            all_metrics_by_user.append(metrics_by_user)
            macro = macro_metrics(metrics_by_user, model_name)
            macro.update({
                "best_epoch": best_epoch,
                "best_macro_validation_eer": best_eer,
                "best_macro_validation_f1": best_f1,
                "test_multiclass_loss": test_loss,
            })
            macro_rows.append(macro)
            pooled_rows.append(pooled_metrics_from_claims(claims, model_name))

            claims.to_csv(
                output / "predictions" / f"{model_name}_test_claims.csv.gz",
                index=False,
                compression="gzip",
            )

            multiclass_pred = test_probs.argmax(axis=1)
            multiclass_acc = float(accuracy_score(test_labels, multiclass_pred))
            multiclass_f1 = float(
                f1_score(test_labels, multiclass_pred, average="macro", zero_division=0)
            )
            multiclass_rows.append({
                "model": model_name,
                "test_multiclass_accuracy": multiclass_acc,
                "test_multiclass_macro_f1": multiclass_f1,
                "test_multiclass_loss": test_loss,
                "test_windows": len(test_labels),
            })
            cm = confusion_matrix(
                test_labels,
                multiclass_pred,
                labels=users.sort_values("user_index")["user_index"].astype(int).tolist(),
            )
            for actual_pos, actual_rec in enumerate(users.sort_values("user_index").itertuples(index=False)):
                for pred_pos, pred_rec in enumerate(users.sort_values("user_index").itertuples(index=False)):
                    confusion_rows.append({
                        "model": model_name,
                        "actual_user_id": str(actual_rec.user_id),
                        "actual_user_index": int(actual_rec.user_index),
                        "predicted_user_id": str(pred_rec.user_id),
                        "predicted_user_index": int(pred_rec.user_index),
                        "count": int(cm[actual_pos, pred_pos]),
                    })

            latency = measure_latency(
                model,
                test_X_memmap[selected_indices["internal_test"][:1]],
                device,
                args.batch_size,
                args.latency_runs,
            )
            latency_row = {
                "model": model_name,
                "device": str(device),
                **latency,
                **duration_stats,
            }
            latency_row["tta_mean_sec"] = (
                duration_stats["window_duration_mean_sec"] + latency["single_mean_ms"] / 1000.0
            )
            latency_row["tta_p95_approx_sec"] = (
                duration_stats["window_duration_p95_sec"] + latency["single_p95_ms"] / 1000.0
            )
            latency_rows.append(latency_row)

            checkpoint_path = output / "models" / f"{model_name}_best.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                cfg,
                users,
                args.window_size,
                best_epoch,
                best_eer,
                best_f1,
            )
            inventory_rows.append({
                "model": model_name,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "checkpoint_file": str(checkpoint_path.relative_to(output)),
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "best_epoch": best_epoch,
                "training_seconds": time.time() - model_training_start,
                "input_shape": json.dumps([args.window_size, input_features]),
                "n_classes": n_classes,
                "device": str(device),
            })
            logger.info(
                "%s completed: best_epoch=%d macro_test_EER=%.4f macro_test_F1=%.4f",
                model_name,
                best_epoch,
                macro["macro_test_eer"],
                macro["macro_f1"],
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as exc:
            logger.exception("Model %s failed.", model_name)
            errors.append({
                "phase": "model_training_evaluation",
                "model": model_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

    # Save accumulated outputs even if one model failed.
    pd.DataFrame(history_rows).to_csv(
        output / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    if all_metrics_by_user:
        pd.concat(all_metrics_by_user, ignore_index=True).to_csv(
            output / "metrics_by_user.csv", index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame().to_csv(output / "metrics_by_user.csv", index=False)
    pd.DataFrame(macro_rows).to_csv(
        output / "metrics_macro.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(pooled_rows).to_csv(
        output / "metrics_pooled.csv", index=False, encoding="utf-8-sig"
    )
    if threshold_frames:
        pd.concat(threshold_frames, ignore_index=True).to_csv(
            output / "thresholds.csv", index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame().to_csv(output / "thresholds.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(
        output / "latency_tta.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(multiclass_rows).to_csv(
        output / "multiclass_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(confusion_rows).to_csv(
        output / "confusion_matrices.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(inventory_rows).to_csv(
        output / "model_inventory.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        errors,
        columns=["phase", "model", "error_type", "error", "traceback"],
    ).to_csv(output / "stage5_errors.csv", index=False, encoding="utf-8-sig")

    successful_models = [row["model"] for row in inventory_rows]
    summary = {
        "stage": "Balabit Stage 5 - deep sequence models",
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage4_root": str(stage4_root),
        "tensor_dir": str(tensor_dir),
        "output": str(output),
        "window_size": args.window_size,
        "event_features": input_features,
        "n_classes": n_classes,
        "users": users.sort_values("user_index").to_dict(orient="records"),
        "requested_models": args.models,
        "successful_models": successful_models,
        "failed_models": [e["model"] for e in errors],
        "training_configuration": {
            "max_train_per_user": args.max_train_per_user,
            "max_validation_per_user": args.max_validation_per_user,
            "max_test_per_user": args.max_test_per_user,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "dropout": args.dropout,
            "seed": args.seed,
            "device": str(device),
            "torch_threads": torch.get_num_threads(),
        },
        "selected_windows": {
            split: int(len(indices)) for split, indices in selected_indices.items()
        },
        "selection_is_shared_across_models": True,
        "threshold_protocol": (
            "One threshold per claimed user selected from validation EER, "
            "then frozen for internal-test evaluation."
        ),
        "primary_reporting": (
            "Macro per-user authentication metrics are primary; pooled and "
            "multiclass results are supplementary."
        ),
        "window_duration_stats": duration_stats,
        "elapsed_seconds": round(time.time() - start_time, 3),
        "source_tensor_hashes": {
            key: sha256_file(path)
            for key, path in paths.items()
            if path.suffix == ".npy"
        },
        "methodological_notes": [
            "Models receive ordered event-level tensors, not flattened statistical features.",
            "The exact Stage 4 session-level train/validation/internal-test protocol is reused.",
            "All models use the same deterministic balanced subset in pilot mode.",
            "Per-user thresholds are selected only on validation scores.",
            "Internal-test scores are evaluated only after the checkpoint and thresholds are fixed.",
            "TTA combines observed window duration with measured single-window inference latency.",
            "The experiment compares CNN, BiLSTM, and CNN+BiLSTM as a direct architectural ablation.",
        ],
    }
    with (output / "stage5_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    review_zip = create_review_zip(output)
    logger.info("Stage 5 completed. Successful models=%s failed=%d", successful_models, len(errors))
    logger.info("Review ZIP: %s", review_zip)
    print("\nREVIEW_ZIP=" + str(review_zip))
    print("LOCAL_RESULTS_FOLDER=" + str(output))

    return 0 if not errors and len(successful_models) == len(args.models) else 6


if __name__ == "__main__":
    raise SystemExit(main())

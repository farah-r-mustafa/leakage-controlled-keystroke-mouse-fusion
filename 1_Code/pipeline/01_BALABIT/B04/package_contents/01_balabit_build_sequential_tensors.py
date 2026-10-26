#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balabit Stage 4 — Sequential Tensor Builder
============================================

Purpose
-------
Build leakage-safe event-level tensors for CNN, BiLSTM, and CNN+BiLSTM while
preserving the exact session-level protocol established in Balabit Stage 2.

Key guarantees
--------------
- Reads the Stage 2 protocol_manifest.csv; it does not create a new split.
- Windows never cross session boundaries.
- Preprocessing parameters are fitted on TRAIN events only.
- Validation and internal-test data never influence clipping/scaling.
- fake_data and official unlabeled test files are not used.
- User identifiers and session identifiers are metadata, never input features.
- Source files are read-only and are never changed.

Outputs
-------
output/
  tensors/
    w30/
      train_X.npy
      train_y_user.npy
      train_session_index.npy
      train_window_index.npy
      validation_X.npy ...
      internal_test_X.npy ...
      train_metadata.csv.gz ...
    w50/ ...
  event_preprocessor.npz
  feature_schema.csv
  users.csv
  sessions.csv
  tensor_counts.csv
  tensor_manifest.csv
  protocol_audit.csv
  stage4_summary.json
  stage4_errors.csv
  run_log.txt
  <output_name>_review.zip

Python 3.10+
Dependencies: pandas, numpy
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import platform
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

VERSION = "1.0.0"
EPS = 1e-9
VALID_SPLITS = ("train", "validation", "internal_test")

CONTINUOUS_FEATURES = [
    "dt",
    "dx",
    "dy",
    "distance",
    "speed",
    "acceleration",
    "jerk",
    "angular_velocity",
    "x",
    "y",
    "sin_angle",
    "cos_angle",
]

BINARY_FEATURES = [
    "is_move",
    "is_drag",
    "is_pressed",
    "is_released",
    "button_none",
    "button_left",
    "button_right",
    "button_middle",
    "is_idle_gap",
]

EVENT_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES

COLUMN_ALIASES = {
    "record_timestamp": {"record_timestamp", "record_time", "server_timestamp"},
    "client_timestamp": {"client_timestamp", "client_time", "timestamp", "time"},
    "button": {"button", "mouse_button", "btn"},
    "state": {"state", "event", "event_type", "action", "type"},
    "x": {"x", "mouse_x", "cursor_x", "x_coord", "x_coordinate"},
    "y": {"y", "mouse_y", "cursor_y", "y_coord", "y_coordinate"},
}


@dataclass(frozen=True)
class ProtocolSession:
    user_id: str
    session_id: str
    relative_file: str
    protocol_split: str
    source_group: str


@dataclass
class Preprocessor:
    clip_low: np.ndarray
    clip_high: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    continuous_features: list[str]
    binary_features: list[str]

    def transform(self, event_df: pd.DataFrame) -> np.ndarray:
        cont = event_df[self.continuous_features].to_numpy(dtype=np.float64, copy=True)
        cont = np.nan_to_num(cont, nan=0.0, posinf=0.0, neginf=0.0)
        cont = np.clip(cont, self.clip_low, self.clip_high)
        cont = (cont - self.mean) / self.scale
        binary = event_df[self.binary_features].to_numpy(dtype=np.float64, copy=False)
        out = np.concatenate([cont, binary], axis=1)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize_column(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-./\\]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def setup_logging(output: Path) -> logging.Logger:
    logger = logging.getLogger("balabit_stage4")
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


def parse_sizes(text: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not vals or any(v < 5 for v in vals):
        raise argparse.ArgumentTypeError("Use comma-separated window sizes >= 5.")
    return vals


def parse_users(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_quantiles(text: str) -> tuple[float, float]:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 2 or not (0.0 <= parts[0] < parts[1] <= 1.0):
        raise argparse.ArgumentTypeError("Use two quantiles such as 0.005,0.995")
    return parts[0], parts[1]


def map_columns(columns: Iterable[str]) -> dict[str, Optional[str]]:
    normalized = {normalize_column(c): str(c) for c in columns}
    out: dict[str, Optional[str]] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        out[canonical] = next(
            (normalized[normalize_column(a)] for a in aliases if normalize_column(a) in normalized),
            None,
        )
    return out


def robust_read_csv(path: Path) -> pd.DataFrame:
    last: Optional[Exception] = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError as exc:
            last = exc
        except Exception as exc:
            last = exc
            break
    raise RuntimeError(f"Could not read {path}: {last}")


def detect_time_scale(values: pd.Series, requested: str) -> tuple[float, str]:
    v = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    if len(v) < 2:
        return 1.0, "seconds_assumed"
    d = np.diff(v)
    pos = d[np.isfinite(d) & (d > 0)]
    median_dt = float(np.median(pos)) if len(pos) else float("nan")
    magnitude = float(np.median(np.abs(v)))

    if requested == "seconds":
        return 1.0, "seconds_forced"
    if requested == "milliseconds":
        return 1e-3, "milliseconds_forced"
    if requested == "microseconds":
        return 1e-6, "microseconds_forced"

    if magnitude >= 1e14:
        return 1e-6, "microseconds_epoch_auto"
    if magnitude >= 1e11:
        return 1e-3, "milliseconds_epoch_auto"
    if magnitude >= 1e8:
        return 1.0, "seconds_epoch_auto"
    if np.isfinite(median_dt):
        if median_dt >= 1000:
            return 1e-6, "microseconds_relative_auto"
        if median_dt >= 1.0:
            return 1e-3, "milliseconds_relative_auto"
    return 1.0, "seconds_relative_auto"


def resolve_source_file(data_root: Path, relative_file: str) -> Path:
    normalized = relative_file.replace("\\", os.sep).replace("/", os.sep)
    candidate = data_root / normalized
    if candidate.is_file():
        return candidate

    # Fallback for a moved dataset root: locate by final training_files/user/session suffix.
    parts = [p for p in Path(normalized).parts if p not in ("", ".")]
    for anchor in ("training_files", "test_files", "fake_data"):
        if anchor in [p.lower() for p in parts]:
            idx = [p.lower() for p in parts].index(anchor)
            suffix = Path(*parts[idx:])
            candidate = data_root / suffix
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Protocol file not found under data root: {relative_file}")


def load_protocol(stage2_root: Path, users: list[str]) -> pd.DataFrame:
    protocol_path = stage2_root / "protocol_manifest.csv"
    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"protocol_manifest.csv not found in Stage 2 folder: {stage2_root}"
        )
    df = pd.read_csv(protocol_path, encoding="utf-8-sig")
    required = {"user_id", "session_id", "file", "protocol_split", "source_group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Stage 2 protocol manifest is missing columns: {sorted(missing)}")

    df = df[df["protocol_split"].isin(VALID_SPLITS)].copy()
    df = df[df["source_group"].eq("training_files")].copy()
    if users:
        unknown = sorted(set(users) - set(df["user_id"].astype(str)))
        if unknown:
            raise ValueError(f"Requested users are not in the protocol manifest: {unknown}")
        df = df[df["user_id"].astype(str).isin(users)].copy()

    if df.empty:
        raise ValueError("No eligible Stage 2 protocol sessions remain after filtering.")

    dupes = df.duplicated(["user_id", "session_id"], keep=False)
    if dupes.any():
        rows = df.loc[dupes, ["user_id", "session_id", "protocol_split"]]
        raise ValueError(f"Duplicate user/session rows in protocol manifes<local_path>")

    return df.sort_values(["user_id", "protocol_split", "session_id"]).reset_index(drop=True)


def prepare_session(
    path: Path,
    timestamp_unit: str,
    remove_exact_duplicates: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = robust_read_csv(path)
    mapping = map_columns(raw.columns)
    required = ["client_timestamp", "x", "y", "state", "button"]
    missing = [c for c in required if mapping.get(c) is None]
    if missing:
        raise ValueError(f"Missing columns {missing}; actual={list(raw.columns)}")

    df = pd.DataFrame({
        "client_timestamp_raw": pd.to_numeric(raw[mapping["client_timestamp"]], errors="coerce"),
        "button": raw[mapping["button"]].astype(str).str.strip(),
        "state": raw[mapping["state"]].astype(str).str.strip(),
        "x": pd.to_numeric(raw[mapping["x"]], errors="coerce"),
        "y": pd.to_numeric(raw[mapping["y"]], errors="coerce"),
    })
    original_rows = len(df)
    df = df.dropna(subset=["client_timestamp_raw", "x", "y"]).copy()
    dropped_invalid = original_rows - len(df)
    exact_duplicates = int(df.duplicated().sum())
    if remove_exact_duplicates:
        df = df.drop_duplicates().copy()

    df["_original_order"] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values(
        ["client_timestamp_raw", "_original_order"],
        kind="mergesort",
    ).reset_index(drop=True)

    scale, unit_used = detect_time_scale(df["client_timestamp_raw"], timestamp_unit)
    t0 = float(df["client_timestamp_raw"].iloc[0]) if len(df) else 0.0
    df["t"] = (df["client_timestamp_raw"] - t0).astype(float) * scale
    df["dt"] = df["t"].diff().fillna(0.0).clip(lower=0.0)
    df["dx"] = df["x"].diff().fillna(0.0)
    df["dy"] = df["y"].diff().fillna(0.0)
    df["distance"] = np.hypot(df["dx"], df["dy"])

    valid_dt = df["dt"] > EPS
    df["speed"] = 0.0
    df.loc[valid_dt, "speed"] = (
        df.loc[valid_dt, "distance"] / df.loc[valid_dt, "dt"]
    )
    speed_diff = df["speed"].diff().fillna(0.0)
    df["acceleration"] = 0.0
    df.loc[valid_dt, "acceleration"] = (
        speed_diff.loc[valid_dt] / df.loc[valid_dt, "dt"]
    )
    acceleration_diff = df["acceleration"].diff().fillna(0.0)
    df["jerk"] = 0.0
    df.loc[valid_dt, "jerk"] = (
        acceleration_diff.loc[valid_dt] / df.loc[valid_dt, "dt"]
    )

    angle = np.arctan2(df["dy"], df["dx"])
    angle_delta = np.arctan2(
        np.sin(angle.diff().fillna(0.0)),
        np.cos(angle.diff().fillna(0.0)),
    )
    df["angular_velocity"] = 0.0
    df.loc[valid_dt, "angular_velocity"] = (
        angle_delta.loc[valid_dt] / df.loc[valid_dt, "dt"]
    )
    df["sin_angle"] = np.sin(angle)
    df["cos_angle"] = np.cos(angle)

    state = df["state"].str.lower().str.strip()
    button = df["button"].str.lower().str.strip()
    df["is_move"] = state.eq("move").astype(np.float32)
    df["is_drag"] = state.eq("drag").astype(np.float32)
    df["is_pressed"] = state.eq("pressed").astype(np.float32)
    df["is_released"] = state.eq("released").astype(np.float32)
    df["button_none"] = button.isin(
        {"nobutton", "no button", "none", "nan"}
    ).astype(np.float32)
    df["button_left"] = button.str.contains("left", na=False).astype(np.float32)
    df["button_right"] = button.str.contains("right", na=False).astype(np.float32)
    df["button_middle"] = button.str.contains(
        "middle|center", regex=True, na=False
    ).astype(np.float32)
    df["is_idle_gap"] = (df["dt"] > 0.5).astype(np.float32)

    for col in CONTINUOUS_FEATURES:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float64)
        )

    duration = float(df["t"].iloc[-1] - df["t"].iloc[0]) if len(df) > 1 else 0.0
    quality = {
        "raw_rows": original_rows,
        "clean_rows": len(df),
        "dropped_invalid_rows": dropped_invalid,
        "exact_duplicates_detected": exact_duplicates,
        "exact_duplicates_removed": exact_duplicates if remove_exact_duplicates else 0,
        "timestamp_unit_used": unit_used,
        "duration_seconds": duration,
        "event_rate_hz": len(df) / duration if duration > 0 else np.nan,
    }
    return df, quality


def session_window_count(n_events: int, window_size: int, overlap: float) -> int:
    if n_events < window_size:
        return 0
    stride = max(1, int(round(window_size * (1.0 - overlap))))
    return ((n_events - window_size) // stride) + 1


def deterministic_training_sample(arr: np.ndarray, per_session_cap: int) -> np.ndarray:
    if len(arr) <= per_session_cap:
        return arr
    indices = np.linspace(0, len(arr) - 1, num=per_session_cap, dtype=np.int64)
    return arr[indices]


def fit_preprocessor(
    training_samples: list[np.ndarray],
    quantiles: tuple[float, float],
) -> Preprocessor:
    if not training_samples:
        raise ValueError("No training events were available for preprocessing.")
    sample = np.concatenate(training_samples, axis=0).astype(np.float64, copy=False)
    sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)

    low = np.quantile(sample, quantiles[0], axis=0)
    high = np.quantile(sample, quantiles[1], axis=0)
    invalid = high <= low
    high[invalid] = low[invalid] + 1.0

    clipped = np.clip(sample, low, high)
    mean = clipped.mean(axis=0)
    scale = clipped.std(axis=0)
    scale[~np.isfinite(scale) | (scale < EPS)] = 1.0

    return Preprocessor(
        clip_low=low.astype(np.float64),
        clip_high=high.astype(np.float64),
        mean=mean.astype(np.float64),
        scale=scale.astype(np.float64),
        continuous_features=list(CONTINUOUS_FEATURES),
        binary_features=list(BINARY_FEATURES),
    )


def save_preprocessor(preprocessor: Preprocessor, path: Path, quantiles: tuple[float, float]) -> None:
    np.savez(
        path,
        clip_low=preprocessor.clip_low,
        clip_high=preprocessor.clip_high,
        mean=preprocessor.mean,
        scale=preprocessor.scale,
        continuous_features=np.array(preprocessor.continuous_features, dtype="U64"),
        binary_features=np.array(preprocessor.binary_features, dtype="U64"),
        all_features=np.array(EVENT_FEATURES, dtype="U64"),
        quantile_low=np.array([quantiles[0]], dtype=np.float64),
        quantile_high=np.array([quantiles[1]], dtype=np.float64),
    )


def feature_schema_df(preprocessor: Preprocessor) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(EVENT_FEATURES):
        is_cont = name in CONTINUOUS_FEATURES
        rows.append({
            "feature_index": idx,
            "feature": name,
            "feature_type": "continuous_scaled" if is_cont else "binary_unscaled",
            "clip_low": (
                float(preprocessor.clip_low[CONTINUOUS_FEATURES.index(name)])
                if is_cont else np.nan
            ),
            "clip_high": (
                float(preprocessor.clip_high[CONTINUOUS_FEATURES.index(name)])
                if is_cont else np.nan
            ),
            "train_mean_after_clipping": (
                float(preprocessor.mean[CONTINUOUS_FEATURES.index(name)])
                if is_cont else np.nan
            ),
            "train_scale_after_clipping": (
                float(preprocessor.scale[CONTINUOUS_FEATURES.index(name)])
                if is_cont else np.nan
            ),
            "input_to_model": True,
            "notes": (
                "Clipped and standardized using training events only"
                if is_cont else "Binary indicator; not standardized"
            ),
        })
    return pd.DataFrame(rows)


def open_output_arrays(
    root: Path,
    window_sizes: list[int],
    counts: dict[tuple[int, str], int],
    n_features: int,
) -> dict[tuple[int, str, str], np.memmap]:
    arrays: dict[tuple[int, str, str], np.memmap] = {}
    for ws in window_sizes:
        wdir = root / f"w{ws}"
        wdir.mkdir(parents=True, exist_ok=True)
        for split in VALID_SPLITS:
            n = counts.get((ws, split), 0)
            arrays[(ws, split, "X")] = np.lib.format.open_memmap(
                wdir / f"{split}_X.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n, ws, n_features),
            )
            arrays[(ws, split, "y_user")] = np.lib.format.open_memmap(
                wdir / f"{split}_y_user.npy",
                mode="w+",
                dtype=np.int16,
                shape=(n,),
            )
            arrays[(ws, split, "session_index")] = np.lib.format.open_memmap(
                wdir / f"{split}_session_index.npy",
                mode="w+",
                dtype=np.int32,
                shape=(n,),
            )
            arrays[(ws, split, "window_index")] = np.lib.format.open_memmap(
                wdir / f"{split}_window_index.npy",
                mode="w+",
                dtype=np.int32,
                shape=(n,),
            )
    return arrays


def create_sequence_view(events: np.ndarray, ws: int, overlap: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(events)
    if n < ws:
        return np.empty((0, ws, events.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64)
    stride = max(1, int(round(ws * (1.0 - overlap))))
    starts = np.arange(0, n - ws + 1, stride, dtype=np.int64)
    view = np.lib.stride_tricks.sliding_window_view(events, ws, axis=0)
    # sliding_window_view with axis=0 gives: (N-ws+1, features, ws)
    view = np.transpose(view[::stride], (0, 2, 1))
    return view, starts


def append_metadata(
    path: Path,
    rows: pd.DataFrame,
    initialized: set[str],
) -> None:
    key = str(path)
    mode = "at" if key in initialized else "wt"
    with gzip.open(path, mode, encoding="utf-8", newline="") as fh:
        rows.to_csv(
            fh,
            index=False,
            header=(key not in initialized),
            lineterminator="\n",
        )
    initialized.add(key)


def audit_protocol(protocol: pd.DataFrame) -> pd.DataFrame:
    audit_rows: list[dict[str, Any]] = []

    duplicate_session = protocol.duplicated(["user_id", "session_id"]).any()
    audit_rows.append({
        "check": "unique_user_session",
        "passed": not duplicate_session,
        "details": "Each user/session occurs once in protocol manifest.",
    })

    cross_split = (
        protocol.groupby(["user_id", "session_id"])["protocol_split"]
        .nunique()
        .gt(1)
        .any()
    )
    audit_rows.append({
        "check": "no_session_cross_split",
        "passed": not cross_split,
        "details": "No session belongs to more than one protocol split.",
    })

    user_split = protocol.groupby("user_id")["protocol_split"].agg(set)
    missing = {
        user: sorted(set(VALID_SPLITS) - set(splits))
        for user, splits in user_split.items()
        if set(VALID_SPLITS) - set(splits)
    }
    audit_rows.append({
        "check": "all_users_in_all_splits",
        "passed": not bool(missing),
        "details": json.dumps(missing, ensure_ascii=False) if missing else "All users occur in train, validation, and internal_test.",
    })

    source_ok = protocol["source_group"].eq("training_files").all()
    audit_rows.append({
        "check": "training_files_only",
        "passed": bool(source_ok),
        "details": "fake_data and official unlabeled test files are excluded.",
    })

    audit_rows.append({
        "check": "preprocessor_train_only",
        "passed": True,
        "details": "Clipping and scaling are fitted exclusively from protocol_split=train events.",
    })

    audit_rows.append({
        "check": "identifiers_excluded_from_features",
        "passed": not any(x in EVENT_FEATURES for x in ("user_id", "session_id", "window_index")),
        "details": "User/session/window identifiers are metadata only.",
    })

    return pd.DataFrame(audit_rows)


def file_manifest_rows(output: Path, tensors_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(tensors_root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(output)
        record: dict[str, Any] = {
            "file": str(rel),
            "size_bytes": p.stat().st_size,
            "extension": p.suffix.lower(),
        }
        if p.suffix == ".npy":
            arr = np.load(p, mmap_mode="r")
            record["dtype"] = str(arr.dtype)
            record["shape"] = json.dumps(list(arr.shape))
        else:
            record["dtype"] = None
            record["shape"] = None
        rows.append(record)
    return rows


def create_review_zip(output: Path) -> Path:
    target = output.parent / f"{output.name}_review.zip"
    if target.exists():
        target.unlink()

    include_names = {
        "event_preprocessor.npz",
        "feature_schema.csv",
        "users.csv",
        "sessions.csv",
        "tensor_counts.csv",
        "tensor_manifest.csv",
        "protocol_audit.csv",
        "stage4_summary.json",
        "stage4_errors.csv",
        "session_quality_stage4.csv",
        "run_log.txt",
    }
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(output.iterdir()):
            if p.is_file() and p.name in include_names:
                zf.write(p, arcname=p.name)
    return target


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Balabit Stage 4 sequential tensor builder")
    ap.add_argument("--data-root", required=True, type=Path, help="Raw Balabit dataset root.")
    ap.add_argument("--stage2-root", required=True, type=Path, help="Stage 2 full results folder containing protocol_manifest.csv.")
    ap.add_argument("--output", type=Path, default=Path("balabit_stage4_tensors"))
    ap.add_argument("--window-sizes", type=parse_sizes, default=[30, 50, 70, 90])
    ap.add_argument("--overlap", type=float, default=0.50)
    ap.add_argument("--users", type=parse_users, default=[], help="Optional comma-separated pilot users.")
    ap.add_argument("--timestamp-unit", choices=["auto", "seconds", "milliseconds", "microseconds"], default="auto")
    ap.add_argument("--remove-exact-duplicates", action="store_true")
    ap.add_argument("--clip-quantiles", type=parse_quantiles, default=(0.005, 0.995))
    ap.add_argument("--preprocessor-sample-per-train-session", type=int, default=10_000)
    ap.add_argument("--write-batch-size", type=int, default=2048)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    stage2_root = args.stage2_root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not data_root.is_dir():
        print(f"ERROR: Raw dataset folder not found: {data_root}", file=sys.stderr)
        return 2
    if not stage2_root.is_dir():
        print(f"ERROR: Stage 2 results folder not found: {stage2_root}", file=sys.stderr)
        return 2
    if not 0 <= args.overlap < 1:
        print("ERROR: overlap must be in [0,1).", file=sys.stderr)
        return 2
    if args.preprocessor_sample_per_train_session <= 0:
        print("ERROR: preprocessor sample cap must be positive.", file=sys.stderr)
        return 2
    if args.write_batch_size <= 0:
        print("ERROR: write batch size must be positive.", file=sys.stderr)
        return 2

    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output)
    started = time.time()

    logger.info("Balabit Stage 4 v%s", VERSION)
    logger.info("Raw data root: %s", data_root)
    logger.info("Stage 2 root: %s", stage2_root)
    logger.info("Output: %s", output)
    logger.info("Window sizes=%s overlap=%.2f", args.window_sizes, args.overlap)
    logger.info("Pilot users=%s", args.users or "ALL")
    logger.info("Python=%s | pandas=%s | numpy=%s", sys.version.replace("\n", " "), pd.__version__, np.__version__)

    try:
        protocol = load_protocol(stage2_root, args.users)
    except Exception as exc:
        logger.exception("Could not load Stage 2 protocol.")
        return 3

    users = sorted(protocol["user_id"].astype(str).unique(), key=lambda x: int(re.sub(r"\D", "", x) or 0))
    user_to_index = {u: i for i, u in enumerate(users)}
    users_df = pd.DataFrame({
        "user_index": [user_to_index[u] for u in users],
        "user_id": users,
    })
    users_df.to_csv(output / "users.csv", index=False, encoding="utf-8-sig")

    session_rows: list[dict[str, Any]] = []
    for idx, row in protocol.iterrows():
        source_path = resolve_source_file(data_root, str(row["file"]))
        session_rows.append({
            "session_index": idx,
            "user_id": str(row["user_id"]),
            "user_index": user_to_index[str(row["user_id"])],
            "session_id": str(row["session_id"]),
            "protocol_split": str(row["protocol_split"]),
            "relative_file": str(row["file"]),
            "resolved_file": str(source_path),
            "source_group": str(row["source_group"]),
        })
    sessions_df = pd.DataFrame(session_rows)
    sessions_df.to_csv(output / "sessions.csv", index=False, encoding="utf-8-sig")

    protocol_audit = audit_protocol(protocol)
    protocol_audit.to_csv(output / "protocol_audit.csv", index=False, encoding="utf-8-sig")
    if not protocol_audit["passed"].all():
        logger.error("Protocol audit failed; see protocol_audit.csv")
        return 4

    counts: dict[tuple[int, str], int] = {
        (ws, split): 0 for ws in args.window_sizes for split in VALID_SPLITS
    }
    quality_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    training_samples: list[np.ndarray] = []

    logger.info("Pass 1/2: validate sessions, count windows, fit train-only preprocessing sample.")
    for _, session in sessions_df.iterrows():
        try:
            event_df, quality = prepare_session(
                Path(session["resolved_file"]),
                args.timestamp_unit,
                args.remove_exact_duplicates,
            )
            quality.update({
                "session_index": int(session["session_index"]),
                "user_id": session["user_id"],
                "user_index": int(session["user_index"]),
                "session_id": session["session_id"],
                "protocol_split": session["protocol_split"],
                "relative_file": session["relative_file"],
            })
            quality_rows.append(quality)

            for ws in args.window_sizes:
                counts[(ws, session["protocol_split"])] += session_window_count(
                    len(event_df), ws, args.overlap
                )

            if session["protocol_split"] == "train":
                cont = event_df[CONTINUOUS_FEATURES].to_numpy(dtype=np.float64)
                training_samples.append(
                    deterministic_training_sample(
                        cont,
                        args.preprocessor_sample_per_train_session,
                    )
                )
        except Exception as exc:
            logger.exception("Pass 1 failed for %s", session["relative_file"])
            errors.append({
                "phase": "pass1",
                "session_index": int(session["session_index"]),
                "user_id": session["user_id"],
                "session_id": session["session_id"],
                "protocol_split": session["protocol_split"],
                "relative_file": session["relative_file"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    pd.DataFrame(quality_rows).to_csv(
        output / "session_quality_stage4.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        errors,
        columns=[
            "phase", "session_index", "user_id", "session_id",
            "protocol_split", "relative_file", "error_type", "error",
        ],
    ).to_csv(output / "stage4_errors.csv", index=False, encoding="utf-8-sig")

    if errors:
        logger.error("Pass 1 contained %d failed sessions. Tensor creation stopped.", len(errors))
        return 5

    preprocessor = fit_preprocessor(training_samples, args.clip_quantiles)
    save_preprocessor(preprocessor, output / "event_preprocessor.npz", args.clip_quantiles)
    feature_schema_df(preprocessor).to_csv(
        output / "feature_schema.csv",
        index=False,
        encoding="utf-8-sig",
    )

    count_rows = [
        {
            "window_size": ws,
            "protocol_split": split,
            "windows": counts[(ws, split)],
            "tensor_shape": json.dumps([counts[(ws, split)], ws, len(EVENT_FEATURES)]),
        }
        for ws in args.window_sizes
        for split in VALID_SPLITS
    ]
    pd.DataFrame(count_rows).to_csv(
        output / "tensor_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tensors_root = output / "tensors"
    tensors_root.mkdir(exist_ok=True)
    arrays = open_output_arrays(
        tensors_root,
        args.window_sizes,
        counts,
        len(EVENT_FEATURES),
    )
    offsets = {(ws, split): 0 for ws in args.window_sizes for split in VALID_SPLITS}
    metadata_initialized: set[str] = set()

    logger.info("Pass 2/2: build normalized sequential tensors.")
    for session_number, session in sessions_df.iterrows():
        try:
            event_df, _ = prepare_session(
                Path(session["resolved_file"]),
                args.timestamp_unit,
                args.remove_exact_duplicates,
            )
            events = preprocessor.transform(event_df)
            split = str(session["protocol_split"])
            user_index = int(session["user_index"])
            session_index = int(session["session_index"])

            for ws in args.window_sizes:
                view, starts = create_sequence_view(events, ws, args.overlap)
                n_windows = len(starts)
                if n_windows == 0:
                    continue

                destination_start = offsets[(ws, split)]
                destination_end = destination_start + n_windows
                expected = counts[(ws, split)]
                if destination_end > expected:
                    raise RuntimeError(
                        f"Write overflow for w{ws}/{split}: {destination_end}>{expected}"
                    )

                batch = args.write_batch_size
                for local_start in range(0, n_windows, batch):
                    local_end = min(local_start + batch, n_windows)
                    arrays[(ws, split, "X")][
                        destination_start + local_start: destination_start + local_end
                    ] = view[local_start:local_end]

                arrays[(ws, split, "y_user")][destination_start:destination_end] = user_index
                arrays[(ws, split, "session_index")][destination_start:destination_end] = session_index
                arrays[(ws, split, "window_index")][destination_start:destination_end] = np.arange(
                    n_windows, dtype=np.int32
                )

                stride = max(1, int(round(ws * (1.0 - args.overlap))))
                end_indices = starts + ws - 1
                t = event_df["t"].to_numpy(dtype=np.float64)
                metadata = pd.DataFrame({
                    "row_index": np.arange(destination_start, destination_end, dtype=np.int64),
                    "user_id": np.repeat(session["user_id"], n_windows),
                    "user_index": np.repeat(user_index, n_windows),
                    "session_id": np.repeat(session["session_id"], n_windows),
                    "session_index": np.repeat(session_index, n_windows),
                    "protocol_split": np.repeat(split, n_windows),
                    "window_size": np.repeat(ws, n_windows),
                    "stride": np.repeat(stride, n_windows),
                    "window_index": np.arange(n_windows, dtype=np.int32),
                    "start_event_index": starts,
                    "end_event_index": end_indices,
                    "start_time": t[starts],
                    "end_time": t[end_indices],
                    "window_duration": np.maximum(t[end_indices] - t[starts], 0.0),
                })
                metadata_path = tensors_root / f"w{ws}" / f"{split}_metadata.csv.gz"
                append_metadata(metadata_path, metadata, metadata_initialized)
                offsets[(ws, split)] = destination_end

            logger.info(
                "[%d/%d] Built tensors for %s %s %s",
                session_number + 1,
                len(sessions_df),
                session["user_id"],
                session["session_id"],
                split,
            )
        except Exception as exc:
            logger.exception("Pass 2 failed for %s", session["relative_file"])
            errors.append({
                "phase": "pass2",
                "session_index": int(session["session_index"]),
                "user_id": session["user_id"],
                "session_id": session["session_id"],
                "protocol_split": session["protocol_split"],
                "relative_file": session["relative_file"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            break

    for arr in arrays.values():
        arr.flush()
    arrays.clear()

    pd.DataFrame(
        errors,
        columns=[
            "phase", "session_index", "user_id", "session_id",
            "protocol_split", "relative_file", "error_type", "error",
        ],
    ).to_csv(output / "stage4_errors.csv", index=False, encoding="utf-8-sig")

    offset_mismatches = []
    for key, expected in counts.items():
        actual = offsets[key]
        if actual != expected:
            offset_mismatches.append({
                "window_size": key[0],
                "protocol_split": key[1],
                "expected": expected,
                "actual": actual,
            })

    tensor_manifest = pd.DataFrame(file_manifest_rows(output, tensors_root))
    tensor_manifest.to_csv(
        output / "tensor_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "stage": "Balabit Stage 4 - sequential tensors",
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "stage2_root": str(stage2_root),
        "output": str(output),
        "users": users,
        "user_count": len(users),
        "sessions": len(sessions_df),
        "window_sizes": args.window_sizes,
        "overlap": args.overlap,
        "event_feature_count": len(EVENT_FEATURES),
        "continuous_features": CONTINUOUS_FEATURES,
        "binary_features": BINARY_FEATURES,
        "clip_quantiles": list(args.clip_quantiles),
        "preprocessor_sample_rows": int(sum(len(x) for x in training_samples)),
        "preprocessing": {
            "continuous": "train-only quantile clipping then StandardScaler-equivalent z-score",
            "binary": "unscaled 0/1 indicators",
            "identifiers_in_input": False,
            "absolute_timestamp_in_input": False,
        },
        "counts": count_rows,
        "offset_mismatches": offset_mismatches,
        "sessions_failed": len(errors),
        "tensor_files_total_bytes": int(tensor_manifest["size_bytes"].sum()) if not tensor_manifest.empty else 0,
        "elapsed_seconds": round(time.time() - started, 3),
        "methodological_notes": [
            "The Stage 2 session split is reused exactly; no new random split is created.",
            "Windows do not cross session boundaries.",
            "Clipping and scaling are fitted using training events only.",
            "Validation and internal-test events do not affect preprocessing.",
            "User and session identifiers are stored only as labels/metadata.",
            "fake_data remains reserved for later mimicry/robustness experiments.",
            "These tensors are intended for CNN, BiLSTM, and CNN+BiLSTM ablation experiments.",
        ],
    }
    with (output / "stage4_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    review_zip = create_review_zip(output)
    logger.info("Tensor building completed.")
    logger.info("Tensor bytes: %d", summary["tensor_files_total_bytes"])
    logger.info("Review ZIP: %s", review_zip)

    print("\nTENSOR_FOLDER=" + str(tensors_root))
    print("REVIEW_ZIP=" + str(review_zip))

    if errors or offset_mismatches:
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balabit Stage 2: Canonical Manifest, Session-Level Protocol, and Window Features
Version 1.1

Read-only preprocessing. It does not train models and never modifies source files.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
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

VERSION = "1.1.0"
EPS = 1e-9


def normalize_column(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-./\\]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


COLUMN_ALIASES = {
    "record_timestamp": {"record_timestamp", "record_time", "server_timestamp"},
    "client_timestamp": {"client_timestamp", "client_time", "timestamp", "time"},
    "button": {"button", "mouse_button", "btn"},
    "state": {"state", "event", "event_type", "action", "type"},
    "x": {"x", "mouse_x", "cursor_x", "x_coord", "x_coordinate"},
    "y": {"y", "mouse_y", "cursor_y", "y_coord", "y_coordinate"},
}


@dataclass
class SessionRef:
    path: Path
    source_group: str
    user_id: Optional[str]
    session_id: str
    raw_split: str


def setup_logging(output: Path) -> logging.Logger:
    logger = logging.getLogger("balabit_stage2")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(output / "run_log.txt", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_user(path: Path) -> Optional[str]:
    for part in path.parts:
        m = re.fullmatch(r"(?i)user[_\- ]?(\d+)", part)
        if m:
            return f"user{int(m.group(1))}"
    m = re.search(r"(?i)(?:^|[^a-z])user[_\- ]?(\d+)", str(path))
    return f"user{int(m.group(1))}" if m else None


def parse_session(path: Path) -> str:
    m = re.search(r"(?i)session[_\- ]?([a-z0-9]+)", path.stem)
    return f"session{m.group(1)}" if m else path.stem


def classify_source(path: Path) -> tuple[str, str]:
    text = str(path).lower()
    if "training_files" in text:
        return "training_files", "genuine_training"
    if "test_files" in text:
        return "test_files", "official_test_unlabeled"
    if "fake_data" in text:
        return "fake_data", "auxiliary_fake"
    return "other", "unknown"


def discover_sessions(root: Path) -> list[SessionRef]:
    refs: list[SessionRef] = []
    for p in sorted(root.rglob("*.csv")):
        if not p.is_file():
            continue
        source_group, raw_split = classify_source(p)
        refs.append(SessionRef(
            path=p,
            source_group=source_group,
            user_id=parse_user(p),
            session_id=parse_session(p),
            raw_split=raw_split,
        ))
    return refs


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def deterministic_rank(user_id: str, session_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{user_id}|{session_id}".encode()).hexdigest()


def assign_session_protocol(training_refs: list[SessionRef], seed: int) -> dict[tuple[str, str], str]:
    by_user: dict[str, list[SessionRef]] = {}
    for ref in training_refs:
        if ref.user_id:
            by_user.setdefault(ref.user_id, []).append(ref)

    assignment: dict[tuple[str, str], str] = {}
    for user, refs in sorted(by_user.items()):
        refs = sorted(refs, key=lambda r: deterministic_rank(user, r.session_id, seed))
        n = len(refs)
        if n < 3:
            raise ValueError(f"{user} has only {n} training sessions; at least 3 are required.")
        n_test = max(1, int(round(n * 0.20)))
        n_val = max(1, int(round(n * 0.20)))
        while n - n_test - n_val < 1:
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
        for i, ref in enumerate(refs):
            split = "internal_test" if i < n_test else "validation" if i < n_test + n_val else "train"
            assignment[(user, ref.session_id)] = split
    return assignment


def map_columns(columns: Iterable[str]) -> dict[str, Optional[str]]:
    normalized = {normalize_column(c): str(c) for c in columns}
    out: dict[str, Optional[str]] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        out[canonical] = next(
            (normalized[normalize_column(a)] for a in aliases if normalize_column(a) in normalized),
            None,
        )
    return out


def detect_time_scale(values: pd.Series, requested: str) -> tuple[float, str, float]:
    v = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    if len(v) < 2:
        return 1.0, "seconds_assumed", float("nan")
    d = np.diff(v)
    pos = d[np.isfinite(d) & (d > 0)]
    median_dt = float(np.median(pos)) if len(pos) else float("nan")
    magnitude = float(np.median(np.abs(v)))
    if requested == "seconds": return 1.0, "seconds_forced", median_dt
    if requested == "milliseconds": return 1e-3, "milliseconds_forced", median_dt
    if requested == "microseconds": return 1e-6, "microseconds_forced", median_dt
    if magnitude >= 1e14: return 1e-6, "microseconds_epoch_auto", median_dt
    if magnitude >= 1e11: return 1e-3, "milliseconds_epoch_auto", median_dt
    if magnitude >= 1e8: return 1.0, "seconds_epoch_auto", median_dt
    if np.isfinite(median_dt):
        if median_dt >= 1000: return 1e-6, "microseconds_relative_auto", median_dt
        if median_dt >= 1.0: return 1e-3, "milliseconds_relative_auto", median_dt
    return 1.0, "seconds_relative_auto", median_dt


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


def prepare_session(ref: SessionRef, timestamp_unit: str, remove_duplicates: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = robust_read_csv(ref.path)
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
    exact_dupes = int(df.duplicated().sum())
    if remove_duplicates:
        df = df.drop_duplicates().copy()

    df["_order"] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values(["client_timestamp_raw", "_order"], kind="mergesort").reset_index(drop=True)
    scale, unit_used, median_raw_dt = detect_time_scale(df["client_timestamp_raw"], timestamp_unit)
    t0 = float(df["client_timestamp_raw"].iloc[0]) if len(df) else 0.0
    df["t"] = (df["client_timestamp_raw"] - t0).astype(float) * scale
    df["dt"] = df["t"].diff().fillna(0.0).clip(lower=0.0)
    df["dx"] = df["x"].diff().fillna(0.0)
    df["dy"] = df["y"].diff().fillna(0.0)
    df["distance"] = np.hypot(df["dx"], df["dy"])
    valid = df["dt"] > EPS
    df["speed"] = 0.0
    df.loc[valid, "speed"] = df.loc[valid, "distance"] / df.loc[valid, "dt"]
    speed_diff = df["speed"].diff().fillna(0.0)
    df["acceleration"] = 0.0
    df.loc[valid, "acceleration"] = speed_diff.loc[valid] / df.loc[valid, "dt"]
    acc_diff = df["acceleration"].diff().fillna(0.0)
    df["jerk"] = 0.0
    df.loc[valid, "jerk"] = acc_diff.loc[valid] / df.loc[valid, "dt"]
    df["angle"] = np.arctan2(df["dy"], df["dx"])
    ad = np.arctan2(np.sin(df["angle"].diff().fillna(0.0)), np.cos(df["angle"].diff().fillna(0.0)))
    df["angular_velocity"] = 0.0
    df.loc[valid, "angular_velocity"] = ad.loc[valid] / df.loc[valid, "dt"]

    s = df["state"].str.lower().str.strip()
    b = df["button"].str.lower().str.strip()
    df["is_move"] = s.eq("move").astype(np.float32)
    df["is_drag"] = s.eq("drag").astype(np.float32)
    df["is_pressed"] = s.eq("pressed").astype(np.float32)
    df["is_released"] = s.eq("released").astype(np.float32)
    df["button_none"] = b.isin({"nobutton", "no button", "none", "nan"}).astype(np.float32)
    df["button_left"] = b.str.contains("left", na=False).astype(np.float32)
    df["button_right"] = b.str.contains("right", na=False).astype(np.float32)
    df["button_middle"] = b.str.contains("middle|center", regex=True, na=False).astype(np.float32)
    df["is_idle_gap"] = (df["dt"] > 0.5).astype(np.float32)

    numeric = ["t", "dt", "dx", "dy", "distance", "speed", "acceleration", "jerk", "angle", "angular_velocity", "x", "y"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)

    duration = float(df["t"].iloc[-1] - df["t"].iloc[0]) if len(df) > 1 else 0.0
    quality = {
        "user_id": ref.user_id,
        "session_id": ref.session_id,
        "source_group": ref.source_group,
        "raw_rows": original_rows,
        "clean_rows": len(df),
        "dropped_invalid_rows": dropped_invalid,
        "exact_duplicates_detected": exact_dupes,
        "exact_duplicates_removed": exact_dupes if remove_duplicates else 0,
        "timestamp_unit_used": unit_used,
        "median_positive_raw_dt": median_raw_dt,
        "duration_seconds": duration,
        "event_rate_hz": len(df) / duration if duration > 0 else np.nan,
        "x_min": float(df["x"].min()) if len(df) else np.nan,
        "x_max": float(df["x"].max()) if len(df) else np.nan,
        "y_min": float(df["y"].min()) if len(df) else np.nan,
        "y_max": float(df["y"].max()) if len(df) else np.nan,
    }
    return df, quality


STAT_SERIES = ["dt", "dx", "dy", "distance", "speed", "acceleration", "jerk", "angle", "angular_velocity"]
RATIO_SERIES = ["is_move", "is_drag", "is_pressed", "is_released", "button_none", "button_left", "button_right", "button_middle", "is_idle_gap"]


def window_view(arr: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    view = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window_size)
    return view[::stride]


def make_window_features(df: pd.DataFrame, ref: SessionRef, split: str, ws: int, overlap: float) -> pd.DataFrame:
    n = len(df)
    if n < ws:
        return pd.DataFrame()
    stride = max(1, int(round(ws * (1.0 - overlap))))
    starts = np.arange(0, n - ws + 1, stride, dtype=np.int64)
    ends = starts + ws - 1
    rows = len(starts)

    out: dict[str, Any] = {
        "dataset": np.repeat("Balabit Mouse Dynamics", rows),
        "user_id": np.repeat(ref.user_id, rows),
        "session_id": np.repeat(ref.session_id, rows),
        "source_group": np.repeat(ref.source_group, rows),
        "protocol_split": np.repeat(split, rows),
        "window_size": np.repeat(ws, rows),
        "stride": np.repeat(stride, rows),
        "window_index": np.arange(rows, dtype=np.int64),
        "start_event_index": starts,
        "end_event_index": ends,
        "event_count": np.repeat(ws, rows),
    }

    t = df["t"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    out["start_time"] = t[starts]
    out["end_time"] = t[ends]
    duration = np.maximum(t[ends] - t[starts], 0.0)
    out["window_duration"] = duration
    out["event_rate_hz"] = np.divide(ws, duration, out=np.zeros_like(duration), where=duration > EPS)

    distance_windows = window_view(df["distance"].to_numpy(dtype=float), ws, stride)
    path_length = distance_windows.sum(axis=1)
    displacement = np.hypot(x[ends] - x[starts], y[ends] - y[starts])
    out["path_length"] = path_length
    out["displacement"] = displacement
    out["straightness"] = np.divide(displacement, path_length, out=np.zeros_like(displacement), where=path_length > EPS)

    xw = window_view(x, ws, stride)
    yw = window_view(y, ws, stride)
    out["x_range"] = xw.max(axis=1) - xw.min(axis=1)
    out["y_range"] = yw.max(axis=1) - yw.min(axis=1)

    for col in RATIO_SERIES:
        w = window_view(df[col].to_numpy(dtype=float), ws, stride)
        name = col.replace("is_", "") + "_ratio" if col.startswith("is_") else col + "_ratio"
        out[name] = w.mean(axis=1)
    out["click_event_count"] = (
        window_view(df["is_pressed"].to_numpy(dtype=float), ws, stride).sum(axis=1)
        + window_view(df["is_released"].to_numpy(dtype=float), ws, stride).sum(axis=1)
    ).astype(np.int32)

    for col in STAT_SERIES:
        w = window_view(df[col].to_numpy(dtype=float), ws, stride)
        out[f"{col}_mean"] = w.mean(axis=1)
        out[f"{col}_std"] = w.std(axis=1)
        out[f"{col}_min"] = w.min(axis=1)
        out[f"{col}_max"] = w.max(axis=1)
        out[f"{col}_median"] = np.median(w, axis=1)

    return pd.DataFrame(out)


def append_csv_gz(df: pd.DataFrame, path: Path, initialized: set[str]) -> None:
    if df.empty:
        return
    key = str(path)
    mode = "at" if key in initialized else "wt"
    with gzip.open(path, mode, encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False, header=(key not in initialized), lineterminator="\n")
    initialized.add(key)


def feature_dictionary() -> pd.DataFrame:
    rows = [
        ("window_duration", "window", "Elapsed seconds in the window"),
        ("event_rate_hz", "window", "Events per second"),
        ("path_length", "window", "Sum of cursor distances"),
        ("displacement", "window", "First-to-last straight distance"),
        ("straightness", "window", "Displacement/path length"),
        ("x_range", "window", "Horizontal coordinate range"),
        ("y_range", "window", "Vertical coordinate range"),
    ]
    for col in RATIO_SERIES:
        name = col.replace("is_", "") + "_ratio" if col.startswith("is_") else col + "_ratio"
        rows.append((name, "event composition", f"Mean indicator of {col}"))
    for base in STAT_SERIES:
        for stat in ["mean", "std", "min", "max", "median"]:
            rows.append((f"{base}_{stat}", "statistical", f"{stat} of {base}"))
    return pd.DataFrame(rows, columns=["feature", "group", "description"])


def zip_output(output: Path) -> Path:
    target = output.with_suffix(".zip")
    if target.exists(): target.unlink()
    with zipfile.ZipFile(target, "w") as zf:
        for p in sorted(output.rglob("*")):
            if p.is_file():
                compression = zipfile.ZIP_STORED if p.suffix == ".gz" else zipfile.ZIP_DEFLATED
                zf.write(p, arcname=p.relative_to(output.parent), compress_type=compression)
    return target


def parse_sizes(text: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not vals or any(x < 5 for x in vals):
        raise argparse.ArgumentTypeError("Use comma-separated window sizes >= 5")
    return vals


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Balabit Stage 2 preprocessing and windows")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=Path("balabit_stage2_results"))
    ap.add_argument("--window-sizes", type=parse_sizes, default=[30, 50, 70, 90])
    ap.add_argument("--overlap", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timestamp-unit", choices=["auto", "seconds", "milliseconds", "microseconds"], default="auto")
    ap.add_argument("--remove-exact-duplicates", action="store_true")
    ap.add_argument("--max-sessions-per-user", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: Dataset root not found: {root}", file=sys.stderr)
        return 2
    if not 0 <= args.overlap < 1:
        print("ERROR: overlap must be in [0,1)", file=sys.stderr)
        return 2

    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output)
    started = time.time()
    logger.info("Balabit Stage 2 v%s", VERSION)
    logger.info("Data root: %s", root)
    logger.info("Window sizes=%s overlap=%.2f", args.window_sizes, args.overlap)

    refs = discover_sessions(root)
    if not refs:
        logger.error("No CSV files found")
        return 3

    inventory = pd.DataFrame([{
        "file": relative(r.path, root),
        "source_group": r.source_group,
        "raw_split": r.raw_split,
        "user_id_or_claimed_user": r.user_id,
        "session_id": r.session_id,
        "size_bytes": r.path.stat().st_size,
    } for r in refs])
    inventory.to_csv(output / "canonical_manifest_all.csv", index=False, encoding="utf-8-sig")

    training = [r for r in refs if r.source_group == "training_files" and r.user_id]
    official = [r for r in refs if r.source_group == "test_files" and r.user_id]
    fake = [r for r in refs if r.source_group == "fake_data"]
    other = [r for r in refs if r.source_group == "other"]
    logger.info("Discovered training=%d official_test=%d fake=%d other=%d", len(training), len(official), len(fake), len(other))

    if args.max_sessions_per_user > 0:
        selected: list[SessionRef] = []
        for user in sorted({r.user_id for r in training if r.user_id}):
            urs = sorted([r for r in training if r.user_id == user], key=lambda r: deterministic_rank(user or "", r.session_id, args.seed))
            selected.extend(urs[:args.max_sessions_per_user])
        training = selected
        logger.info("Pilot limit: %d sessions selected", len(training))

    protocol = assign_session_protocol(training, args.seed)
    protocol_df = pd.DataFrame([{
        "dataset": "Balabit Mouse Dynamics",
        "user_id": r.user_id,
        "session_id": r.session_id,
        "file": relative(r.path, root),
        "protocol_split": protocol[(r.user_id or "", r.session_id)],
        "source_group": r.source_group,
        "label_semantics": "genuine session of user_id",
    } for r in training])
    protocol_df.to_csv(output / "protocol_manifest.csv", index=False, encoding="utf-8-sig")

    feature_dir = output / "window_features"
    feature_dir.mkdir(exist_ok=True)
    initialized: set[str] = set()
    quality_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    counts: dict[tuple[str, int], int] = {}

    for i, ref in enumerate(training, 1):
        split = protocol[(ref.user_id or "", ref.session_id)]
        logger.info("[%d/%d] %s %s %s", i, len(training), ref.user_id, ref.session_id, split)
        try:
            df, q = prepare_session(ref, args.timestamp_unit, args.remove_exact_duplicates)
            q.update({"file": relative(ref.path, root), "protocol_split": split})
            quality_rows.append(q)
            for ws in args.window_sizes:
                wf = make_window_features(df, ref, split, ws, args.overlap)
                if not wf.empty:
                    path = feature_dir / f"window_features_w{ws}.csv.gz"
                    append_csv_gz(wf, path, initialized)
                    counts[(split, ws)] = counts.get((split, ws), 0) + len(wf)
        except Exception as exc:
            logger.exception("Session failed: %s", ref.path)
            error_rows.append({
                "file": relative(ref.path, root), "user_id": ref.user_id,
                "session_id": ref.session_id, "protocol_split": split,
                "error_type": type(exc).__name__, "error": str(exc),
            })

    pd.DataFrame(quality_rows).to_csv(output / "session_quality.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(error_rows, columns=["file","user_id","session_id","protocol_split","error_type","error"]).to_csv(
        output / "session_errors.csv", index=False, encoding="utf-8-sig"
    )
    feature_dictionary().to_csv(output / "feature_dictionary.csv", index=False, encoding="utf-8-sig")
    count_rows = [{"protocol_split": s, "window_size": w, "windows": n} for (s,w),n in sorted(counts.items())]
    pd.DataFrame(count_rows).to_csv(output / "window_counts.csv", index=False, encoding="utf-8-sig")

    qdf = pd.DataFrame(quality_rows)
    summary = {
        "stage": "Balabit Stage 2 - canonical protocol and statistical windows",
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "window_sizes": args.window_sizes,
        "overlap": args.overlap,
        "seed": args.seed,
        "timestamp_unit_request": args.timestamp_unit,
        "remove_exact_duplicates": args.remove_exact_duplicates,
        "sessions_discovered": {
            "training": len([r for r in refs if r.source_group == "training_files"]),
            "official_test_unlabeled": len(official),
            "fake_excluded": len(fake),
            "other": len(other),
        },
        "training_sessions_selected": len(training),
        "sessions_processed": len(quality_rows),
        "sessions_failed": len(error_rows),
        "clean_rows_total": int(pd.to_numeric(qdf.get("clean_rows"), errors="coerce").fillna(0).sum()) if not qdf.empty else 0,
        "window_counts": count_rows,
        "elapsed_seconds": round(time.time() - started, 3),
        "methodological_notes": [
            "Session-level deterministic split; no row-level leakage.",
            "Windows never cross session boundaries.",
            "fake_data excluded and reserved for robustness/mimicry experiments.",
            "Official test files are inventoried but not used because no authoritative labels were supplied.",
            "No scaler is fitted in this stage.",
            "Statistical windows are stored as compressed CSV files for portability.",
        ],
    }
    with (output / "stage2_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    zip_path = zip_output(output)
    logger.info("Completed processed=%d failed=%d", len(quality_rows), len(error_rows))
    logger.info("ZIP: %s", zip_path)
    print("\nRESULT_ZIP=" + str(zip_path))
    return 0 if not error_rows else 5


if __name__ == "__main__":
    raise SystemExit(main())

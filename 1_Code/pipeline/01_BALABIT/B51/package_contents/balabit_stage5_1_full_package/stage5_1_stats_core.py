#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balabit Stage 5.1 — Statistical correction patch v3.1

Purpose
-------
Recompute only the statistical layer from the full prediction files already
produced by Pilot v3. No model training, validation export, or inference is run.

Corrections
-----------
1. Keep user-level bootstrap over complete claimed-user authentication tasks.
2. Replace the invalid global actual-session bootstrap with a stratified
   session-block bootstrap *within each claimed user*, preserving both genuine
   and impostor classes in every replicate and moving all overlapping windows
   from a sampled session together.
3. Add an exact paired sign-flip permutation test over users.
4. Mark claim-level McNemar as descriptive/sensitivity evidence because claims
   share windows and sessions and therefore are not independent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve
except Exception as exc:
    print("ERROR: numpy, pandas, and scikit-learn are required:", repr(exc))
    print("Run: python -m pip install numpy pandas scikit-learn")
    raise SystemExit(2)

VERSION = "5.1-pilot-v3.1-stats-correction"
SOURCE_OUTPUT_DIR = "balabit_stage5_1_pilot_v3_output"
CORRECTED_OUTPUT_DIR = "balabit_stage5_1_pilot_v3_1_corrected_output"
METRICS = ("FAR", "FRR", "EER", "AUC", "F1", "MCC", "Accuracy")
DELTA_METRICS = ("EER", "MCC", "FAR", "FRR", "AUC")
LOWER_IS_BETTER = {"EER", "FAR", "FRR"}
PAIR_SPECS = {
    "realistic_all_impostors": [("CNN", "HYBRID")],
    "balanced_1to1_common": [
        ("CNN", "HYBRID"),
        ("CNN", "RANDOM_FOREST"),
        ("HYBRID", "RANDOM_FOREST"),
        ("CNN", "LINEAR_SVM"),
        ("HYBRID", "LINEAR_SVM"),
    ],
}
PRED_FILES = {
    ("realistic_all_impostors", "CNN"): "realistic_all_impostors__cnn__predictions.csv.gz",
    ("realistic_all_impostors", "HYBRID"): "realistic_all_impostors__hybrid__predictions.csv.gz",
    ("balanced_1to1_common", "CNN"): "balanced_1to1_common__cnn__predictions.csv.gz",
    ("balanced_1to1_common", "HYBRID"): "balanced_1to1_common__hybrid__predictions.csv.gz",
    ("balanced_1to1_common", "RANDOM_FOREST"): "balanced_1to1_common__random_forest__predictions.csv.gz",
    ("balanced_1to1_common", "LINEAR_SVM"): "balanced_1to1_common__linear_svm__predictions.csv.gz",
}
REQUIRED_COLUMNS = {
    "record_key", "claimed_user_id", "actual_user_id", "session_id",
    "y_true", "score", "y_pred",
}


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_seed(text: str, base: int) -> int:
    d = hashlib.blake2b(f"{base}|{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "little") % (2**32 - 1)


def resolve_project_root(script_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / SOURCE_OUTPUT_DIR / "full_predictions").exists():
            raise FileNotFoundError(root / SOURCE_OUTPUT_DIR / "full_predictions")
        return root

    seeds = [script_dir, script_dir.parent, Path.cwd(), Path.cwd().parent]
    candidates: List[Path] = []
    for seed in seeds:
        try:
            seed = seed.resolve()
        except Exception:
            continue
        candidates.extend([seed, seed.parent, seed.parent.parent])
        for parent in [seed, seed.parent, seed.parent.parent]:
            try:
                candidates.extend([p for p in parent.iterdir() if p.is_dir()])
            except Exception:
                pass
    seen = set()
    for c in candidates:
        try:
            c = c.resolve()
        except Exception:
            continue
        if str(c) in seen:
            continue
        seen.add(str(c))
        if (c / SOURCE_OUTPUT_DIR / "full_predictions").exists():
            return c

    # Bounded recursive fallback.
    for seed in seeds[:3]:
        try:
            base = seed.resolve()
            for p in base.glob(f"**/{SOURCE_OUTPUT_DIR}/full_predictions"):
                if len(p.parts) - len(base.parts) <= 5:
                    return p.parent.parent.resolve()
        except Exception:
            pass
    raise FileNotFoundError(
        f"Could not locate {SOURCE_OUTPUT_DIR}/full_predictions. "
        "Use --project-root to specify the directory containing the v3 output."
    )


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan


def eer_value(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y, score)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def binary_metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = np.asarray(pred, dtype=int)
    pos = y == 1
    neg = y == 0
    far = float(np.mean(pred[neg] == 1)) if neg.any() else np.nan
    frr = float(np.mean(pred[pos] == 0)) if pos.any() else np.nan
    return {
        "FAR": far,
        "FRR": frr,
        "EER": eer_value(y, score),
        "AUC": safe_auc(y, score),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred)) if len(np.unique(y)) == 2 and len(np.unique(pred)) > 1 else 0.0,
        "Accuracy": float(accuracy_score(y, pred)),
    }


def per_user_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for user, g in df.groupby("claimed_user_id", sort=True):
        rows.append({
            "claimed_user_id": user,
            **binary_metrics(g["y_true"].to_numpy(), g["score"].to_numpy(), g["y_pred"].to_numpy()),
        })
    return pd.DataFrame(rows).set_index("claimed_user_id")


def load_predictions(source_out: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    pred_dir = source_out / "full_predictions"
    protocols: Dict[str, Dict[str, pd.DataFrame]] = {}
    for (protocol, model), filename in PRED_FILES.items():
        path = pred_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, low_memory=False)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{filename}: missing columns {sorted(missing)}")
        if df["record_key"].duplicated().any():
            raise ValueError(f"{filename}: duplicate record_key values")
        df["claimed_user_id"] = df["claimed_user_id"].astype(str)
        df["actual_user_id"] = df["actual_user_id"].astype(str)
        df["session_id"] = df["session_id"].astype(str)
        df["y_true"] = pd.to_numeric(df["y_true"], errors="raise").astype(int)
        df["y_pred"] = pd.to_numeric(df["y_pred"], errors="raise").astype(int)
        df["score"] = pd.to_numeric(df["score"], errors="raise").astype(float)
        protocols.setdefault(protocol, {})[model] = df
    return protocols


def labels_and_keys_consistent(models: Mapping[str, pd.DataFrame]) -> None:
    reference_model = next(iter(models))
    ref = models[reference_model][["record_key", "y_true"]].sort_values("record_key").reset_index(drop=True)
    for model, df in models.items():
        cur = df[["record_key", "y_true"]].sort_values("record_key").reset_index(drop=True)
        if len(cur) != len(ref) or not cur["record_key"].equals(ref["record_key"]):
            raise ValueError(f"Record-key mismatch between {reference_model} and {model}")
        if not cur["y_true"].equals(ref["y_true"]):
            raise ValueError(f"Label mismatch between {reference_model} and {model}")


def user_bootstrap(df: pd.DataFrame, reps: int, seed: int) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    pu = per_user_metric_table(df)
    users = pu.index.to_list()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(users), size=(reps, len(users)))
    distributions = {}
    rows = []
    for metric in METRICS:
        vals = pu[metric].to_numpy(dtype=float)
        dist = np.nanmean(vals[draws], axis=1)
        distributions[metric] = dist
        point = float(np.nanmean(vals))
        rows.append({
            "cluster_level": "claimed_user",
            "bootstrap_scheme": "resample_complete_claimed_user_tasks",
            "metric": metric,
            "point_estimate": point,
            "bootstrap_median": float(np.nanmedian(dist)),
            "ci95_low": float(np.nanpercentile(dist, 2.5)),
            "ci95_high": float(np.nanpercentile(dist, 97.5)),
            "replicates": reps,
            "valid_replicates": int(np.isfinite(dist).sum()),
            "all_users_have_both_classes": True,
        })
    return distributions, pd.DataFrame(rows)


def build_stratified_session_blocks(df: pd.DataFrame) -> Tuple[Dict[str, Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]], pd.DataFrame]:
    blocks: Dict[str, Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {}
    inventory_rows = []
    for user, ug in df.groupby("claimed_user_id", sort=True):
        blocks[user] = {0: [], 1: []}
        for label in (0, 1):
            lg = ug[ug["y_true"] == label]
            for (actual_user, session), sg in lg.groupby(["actual_user_id", "session_id"], sort=False):
                y = sg["y_true"].to_numpy(dtype=int)
                score = sg["score"].to_numpy(dtype=float)
                pred = sg["y_pred"].to_numpy(dtype=int)
                if len(y) == 0 or not np.all(y == label):
                    raise ValueError(f"Invalid session block for claimed user {user}, label={label}")
                blocks[user][label].append((y, score, pred))
            inventory_rows.append({
                "claimed_user_id": user,
                "class_label": label,
                "class_name": "genuine" if label == 1 else "impostor",
                "session_blocks": len(blocks[user][label]),
                "claims": len(lg),
            })
        if not blocks[user][0] or not blocks[user][1]:
            raise ValueError(f"Claimed user {user} does not have both genuine and impostor session blocks")
    return blocks, pd.DataFrame(inventory_rows)


def stratified_session_bootstrap(df: pd.DataFrame, reps: int, seed: int) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    blocks, inventory = build_stratified_session_blocks(df)
    users = sorted(blocks)
    rng = np.random.default_rng(seed)
    dist_map = {metric: np.empty(reps, dtype=float) for metric in METRICS}
    complete_reps = 0

    for r in range(reps):
        per_user_values = {metric: [] for metric in METRICS}
        replicate_complete = True
        for user in users:
            pieces_y: List[np.ndarray] = []
            pieces_score: List[np.ndarray] = []
            pieces_pred: List[np.ndarray] = []
            for label in (1, 0):  # explicit genuine then impostor
                group_list = blocks[user][label]
                draw = rng.integers(0, len(group_list), size=len(group_list))
                for idx in draw:
                    y, score, pred = group_list[int(idx)]
                    pieces_y.append(y)
                    pieces_score.append(score)
                    pieces_pred.append(pred)
            y = np.concatenate(pieces_y)
            score = np.concatenate(pieces_score)
            pred = np.concatenate(pieces_pred)
            if len(np.unique(y)) != 2:
                replicate_complete = False
            metrics = binary_metrics(y, score, pred)
            for metric in METRICS:
                per_user_values[metric].append(metrics[metric])
        if replicate_complete:
            complete_reps += 1
        for metric in METRICS:
            dist_map[metric][r] = float(np.nanmean(per_user_values[metric]))

    point_table = per_user_metric_table(df)
    rows = []
    for metric in METRICS:
        arr = dist_map[metric]
        point = float(point_table[metric].mean(skipna=True))
        rows.append({
            "cluster_level": "stratified_actual_session_within_claimed_user",
            "bootstrap_scheme": "resample_complete_session_blocks_within_user_and_class",
            "metric": metric,
            "point_estimate": point,
            "bootstrap_median": float(np.nanmedian(arr)),
            "ci95_low": float(np.nanpercentile(arr, 2.5)),
            "ci95_high": float(np.nanpercentile(arr, 97.5)),
            "replicates": reps,
            "valid_replicates": int(np.isfinite(arr).sum()),
            "all_users_have_both_classes": bool(complete_reps == reps),
        })
    return dist_map, pd.DataFrame(rows), inventory


def bootstrap_ci(df: pd.DataFrame, protocol: str, model: str, user_reps: int, session_reps: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _, user_rows = user_bootstrap(df, user_reps, stable_seed(f"user|{protocol}|{model}", seed))
    _, session_rows, inventory = stratified_session_bootstrap(df, session_reps, stable_seed(f"session|{protocol}|{model}", seed))
    rows = pd.concat([user_rows, session_rows], ignore_index=True)
    rows.insert(0, "model", model)
    rows.insert(0, "protocol", protocol)
    tol = 1e-12
    rows["ci_contains_point"] = (rows["point_estimate"] >= rows["ci95_low"] - tol) & (rows["point_estimate"] <= rows["ci95_high"] + tol)
    inventory.insert(0, "model", model)
    inventory.insert(0, "protocol", protocol)
    return rows, inventory


def aligned_pair(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    aa = a[["record_key", "claimed_user_id", "actual_user_id", "session_id", "y_true", "score", "y_pred"]].rename(
        columns={"score": "score_a", "y_pred": "pred_a"}
    )
    bb = b[["record_key", "y_true", "score", "y_pred"]].rename(
        columns={"y_true": "y_b", "score": "score_b", "y_pred": "pred_b"}
    )
    m = aa.merge(bb, on="record_key", how="inner", validate="one_to_one")
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError("Paired models do not have an exact record-level intersection")
    if not (m["y_true"] == m["y_b"]).all():
        raise ValueError("Paired label conflict")
    return m.drop(columns=["y_b"])


def paired_user_tables(m: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_a, rows_b = [], []
    for user, g in m.groupby("claimed_user_id", sort=True):
        rows_a.append({"claimed_user_id": user, **binary_metrics(g["y_true"].to_numpy(), g["score_a"].to_numpy(), g["pred_a"].to_numpy())})
        rows_b.append({"claimed_user_id": user, **binary_metrics(g["y_true"].to_numpy(), g["score_b"].to_numpy(), g["pred_b"].to_numpy())})
    a = pd.DataFrame(rows_a).set_index("claimed_user_id")
    b = pd.DataFrame(rows_b).set_index("claimed_user_id")
    common = a.index.intersection(b.index)
    return a.loc[common], b.loc[common]


def build_paired_session_blocks(m: pd.DataFrame) -> Dict[str, Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]]:
    blocks: Dict[str, Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]] = {}
    for user, ug in m.groupby("claimed_user_id", sort=True):
        blocks[user] = {0: [], 1: []}
        for label in (0, 1):
            lg = ug[ug["y_true"] == label]
            for _, sg in lg.groupby(["actual_user_id", "session_id"], sort=False):
                blocks[user][label].append((
                    sg["y_true"].to_numpy(dtype=int),
                    sg["score_a"].to_numpy(dtype=float),
                    sg["pred_a"].to_numpy(dtype=int),
                    sg["score_b"].to_numpy(dtype=float),
                    sg["pred_b"].to_numpy(dtype=int),
                ))
        if not blocks[user][0] or not blocks[user][1]:
            raise ValueError(f"Paired claimed user {user} lacks a genuine or impostor session block")
    return blocks


def paired_delta_ci(a: pd.DataFrame, b: pd.DataFrame, model_a: str, model_b: str, protocol: str, user_reps: int, session_reps: int, seed: int) -> pd.DataFrame:
    m = aligned_pair(a, b)
    ua, ub = paired_user_tables(m)
    users = ua.index.to_list()
    rows = []

    rng = np.random.default_rng(stable_seed(f"paired-user|{protocol}|{model_a}|{model_b}", seed))
    draws = rng.integers(0, len(users), size=(user_reps, len(users)))
    for metric in DELTA_METRICS:
        diff = ua[metric].to_numpy(dtype=float) - ub[metric].to_numpy(dtype=float)
        dist = np.nanmean(diff[draws], axis=1)
        point = float(np.nanmean(diff))
        rows.append({
            "protocol": protocol, "model_a": model_a, "model_b": model_b,
            "cluster_level": "claimed_user",
            "bootstrap_scheme": "paired_resample_complete_claimed_user_tasks",
            "metric": f"Delta_{metric}", "definition": "model_a_minus_model_b",
            "point_estimate": point, "bootstrap_median": float(np.nanmedian(dist)),
            "ci95_low": float(np.nanpercentile(dist, 2.5)), "ci95_high": float(np.nanpercentile(dist, 97.5)),
            "replicates": user_reps, "valid_replicates": int(np.isfinite(dist).sum()),
            "all_users_have_both_classes": True,
        })

    blocks = build_paired_session_blocks(m)
    users = sorted(blocks)
    rng = np.random.default_rng(stable_seed(f"paired-session|{protocol}|{model_a}|{model_b}", seed))
    dist_map = {metric: np.empty(session_reps, dtype=float) for metric in DELTA_METRICS}
    complete_reps = 0
    for r in range(session_reps):
        diffs = {metric: [] for metric in DELTA_METRICS}
        complete = True
        for user in users:
            ys: List[np.ndarray] = []
            sa: List[np.ndarray] = []
            pa: List[np.ndarray] = []
            sb: List[np.ndarray] = []
            pb: List[np.ndarray] = []
            for label in (1, 0):
                gl = blocks[user][label]
                draw = rng.integers(0, len(gl), size=len(gl))
                for idx in draw:
                    y, score_a, pred_a, score_b, pred_b = gl[int(idx)]
                    ys.append(y); sa.append(score_a); pa.append(pred_a); sb.append(score_b); pb.append(pred_b)
            y = np.concatenate(ys)
            score_a = np.concatenate(sa); pred_a = np.concatenate(pa)
            score_b = np.concatenate(sb); pred_b = np.concatenate(pb)
            if len(np.unique(y)) != 2:
                complete = False
            ma = binary_metrics(y, score_a, pred_a)
            mb = binary_metrics(y, score_b, pred_b)
            for metric in DELTA_METRICS:
                diffs[metric].append(ma[metric] - mb[metric])
        if complete:
            complete_reps += 1
        for metric in DELTA_METRICS:
            dist_map[metric][r] = float(np.nanmean(diffs[metric]))

    for metric in DELTA_METRICS:
        arr = dist_map[metric]
        point = float(ua[metric].mean(skipna=True) - ub[metric].mean(skipna=True))
        rows.append({
            "protocol": protocol, "model_a": model_a, "model_b": model_b,
            "cluster_level": "stratified_actual_session_within_claimed_user",
            "bootstrap_scheme": "paired_resample_complete_session_blocks_within_user_and_class",
            "metric": f"Delta_{metric}", "definition": "model_a_minus_model_b",
            "point_estimate": point, "bootstrap_median": float(np.nanmedian(arr)),
            "ci95_low": float(np.nanpercentile(arr, 2.5)), "ci95_high": float(np.nanpercentile(arr, 97.5)),
            "replicates": session_reps, "valid_replicates": int(np.isfinite(arr).sum()),
            "all_users_have_both_classes": bool(complete_reps == session_reps),
        })

    out = pd.DataFrame(rows)
    tol = 1e-12
    out["ci_contains_point"] = (out["point_estimate"] >= out["ci95_low"] - tol) & (out["point_estimate"] <= out["ci95_high"] + tol)
    return out


def exact_signflip_p(diff: np.ndarray, seed: int = 2026, monte_carlo_reps: int = 200000) -> Tuple[float, int, str]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[np.abs(diff) > 1e-15]
    n = len(diff)
    if n == 0:
        return 1.0, 0, "all_zero"
    observed = float(np.mean(diff))
    if n <= 20:
        total = 1 << n
        extreme = 0
        for mask in range(total):
            signed_sum = 0.0
            for i, value in enumerate(diff):
                signed_sum += value if ((mask >> i) & 1) else -value
            stat = signed_sum / n
            if abs(stat) >= abs(observed) - 1e-15:
                extreme += 1
        return float(extreme / total), total, "exact_enumeration"
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(monte_carlo_reps, n))
    stats = np.mean(signs * diff[None, :], axis=1)
    p = float((1 + np.sum(np.abs(stats) >= abs(observed))) / (monte_carlo_reps + 1))
    return p, monte_carlo_reps, "monte_carlo"


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted.tolist()


def paired_user_signflip_tests(protocols: Mapping[str, Mapping[str, pd.DataFrame]], seed: int) -> pd.DataFrame:
    rows = []
    for protocol, pairs in PAIR_SPECS.items():
        models = protocols[protocol]
        for a, b in pairs:
            # Keep HYBRID first when comparing against CNN, matching delta outputs.
            model_a, model_b = (("HYBRID", "CNN") if {a, b} == {"CNN", "HYBRID"} else (a, b))
            m = aligned_pair(models[model_a], models[model_b])
            ua, ub = paired_user_tables(m)
            for metric in DELTA_METRICS:
                diff = ua[metric].to_numpy(dtype=float) - ub[metric].to_numpy(dtype=float)
                p, permutations, method = exact_signflip_p(diff, stable_seed(f"signflip|{protocol}|{model_a}|{model_b}|{metric}", seed))
                point = float(np.nanmean(diff))
                direction = (
                    "model_a_better" if ((metric in LOWER_IS_BETTER and point < 0) or (metric not in LOWER_IS_BETTER and point > 0))
                    else "model_b_better" if point != 0 else "tie"
                )
                rows.append({
                    "protocol": protocol,
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": f"Delta_{metric}",
                    "definition": "model_a_minus_model_b",
                    "users": len(diff),
                    "observed_mean_delta": point,
                    "two_sided_p_value": p,
                    "permutations": permutations,
                    "method": method,
                    "direction": direction,
                })
    out = pd.DataFrame(rows)
    out["holm_adjusted_p_within_protocol"] = np.nan
    for protocol, idx in out.groupby("protocol").groups.items():
        out.loc[idx, "holm_adjusted_p_within_protocol"] = holm_adjust(out.loc[idx, "two_sided_p_value"].tolist())
    out["significant_after_holm_0_05"] = out["holm_adjusted_p_within_protocol"] < 0.05
    return out


def corrected_hybrid_justification(metrics: pd.DataFrame, deltas: pd.DataFrame, signflip: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    cnn_cost = costs[costs["model"] == "CNN"].iloc[0]
    hyb_cost = costs[costs["model"] == "HYBRID"].iloc[0]
    ratios = {
        "hybrid_to_cnn_parameter_ratio": float(hyb_cost["total_parameters"] / cnn_cost["total_parameters"]),
        "hybrid_to_cnn_size_ratio": float(hyb_cost["checkpoint_size_mb"] / cnn_cost["checkpoint_size_mb"]),
        "hybrid_to_cnn_latency_ratio": float(hyb_cost["single_mean_ms"] / cnn_cost["single_mean_ms"]),
        "hybrid_to_cnn_throughput_ratio": float(hyb_cost["throughput_windows_per_second"] / cnn_cost["throughput_windows_per_second"]),
    }
    rows = []
    for protocol in ("realistic_all_impostors", "balanced_1to1_common"):
        ms = metrics[(metrics["protocol"] == protocol) & (metrics["aggregation"] == "macro_per_user")].set_index("model")
        dd = deltas[(deltas["protocol"] == protocol) & (deltas["model_a"] == "HYBRID") & (deltas["model_b"] == "CNN") & (deltas["cluster_level"] == "claimed_user")]
        pp = signflip[(signflip["protocol"] == protocol) & (signflip["model_a"] == "HYBRID") & (signflip["model_b"] == "CNN")]
        eer_ci = dd[dd["metric"] == "Delta_EER"].iloc[0]
        mcc_ci = dd[dd["metric"] == "Delta_MCC"].iloc[0]
        eer_perm = pp[pp["metric"] == "Delta_EER"].iloc[0]
        mcc_perm = pp[pp["metric"] == "Delta_MCC"].iloc[0]
        eer_supported = bool(float(eer_ci["ci95_high"]) < 0 and bool(eer_perm["significant_after_holm_0_05"]))
        mcc_supported = bool(float(mcc_ci["ci95_low"]) > 0 and bool(mcc_perm["significant_after_holm_0_05"]))
        primary_supported = eer_supported or mcc_supported
        if primary_supported:
            decision = "STATISTICALLY_SUPPORTED_BUT_COSTLY"
            interpretation = "Hybrid is supported on at least one primary metric by both user-cluster bootstrap and Holm-adjusted user sign-flip testing, but its edge cost remains substantially higher."
        else:
            decision = "NOT_JUSTIFIED_OVER_CNN"
            interpretation = "Hybrid does not establish a robust primary-metric advantage over CNN after cluster-aware inference; CNN is the preferred deployment model, while Hybrid may be reported as a marginal accuracy-oriented alternative."
        rows.append({
            "protocol": protocol,
            "cnn_EER": float(ms.loc["CNN", "EER"]),
            "hybrid_EER": float(ms.loc["HYBRID", "EER"]),
            "Delta_EER_hybrid_minus_cnn": float(ms.loc["HYBRID", "EER"] - ms.loc["CNN", "EER"]),
            "cnn_MCC": float(ms.loc["CNN", "MCC"]),
            "hybrid_MCC": float(ms.loc["HYBRID", "MCC"]),
            "Delta_MCC_hybrid_minus_cnn": float(ms.loc["HYBRID", "MCC"] - ms.loc["CNN", "MCC"]),
            "EER_supported_by_bootstrap_and_signflip": eer_supported,
            "MCC_supported_by_bootstrap_and_signflip": mcc_supported,
            **ratios,
            "decision": decision,
            "interpretation": interpretation,
        })
    return pd.DataFrame(rows)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Correct Stage 5.1 Pilot v3 cluster bootstrap without rerunning models")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--user-replicates", type=int, default=2000)
    parser.add_argument("--session-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    started = time.time()
    script_dir = Path(__file__).resolve().parent
    root = resolve_project_root(script_dir, args.project_root)
    source_out = root / SOURCE_OUTPUT_DIR
    out = root / CORRECTED_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run_console.log"

    class Tee:
        def __init__(self, *streams: Any): self.streams = streams
        def write(self, data: str) -> None:
            for s in self.streams: s.write(data); s.flush()
        def flush(self) -> None:
            for s in self.streams: s.flush()

    stdout0, stderr0 = sys.stdout, sys.stderr
    manifest: Dict[str, Any] = {
        "version": VERSION,
        "status": "RUNNING",
        "project_root": str(root),
        "source_v3_output": str(source_out),
        "source_read_only": True,
        "user_bootstrap_replicates": int(args.user_replicates),
        "session_bootstrap_replicates": int(args.session_replicates),
        "seed": int(args.seed),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with log_path.open("w", encoding="utf-8") as log:
        sys.stdout = Tee(stdout0, log)
        sys.stderr = Tee(stderr0, log)
        try:
            print(f"Balabit Stage 5.1 statistical correction {VERSION}")
            print("Project root:", root)
            print("Source output:", source_out)
            protocols = load_predictions(source_out)
            for protocol, models in protocols.items():
                labels_and_keys_consistent(models)
                print(f"[OK] {protocol}: {', '.join(f'{m}={len(d):,}' for m, d in models.items())}")

            bootstrap_frames: List[pd.DataFrame] = []
            inventory_frames: List[pd.DataFrame] = []
            for protocol, models in protocols.items():
                for model, df in models.items():
                    print(f"[BOOTSTRAP] {protocol} / {model}")
                    b, inv = bootstrap_ci(
                        df, protocol, model,
                        user_reps=max(100, int(args.user_replicates)),
                        session_reps=max(100, int(args.session_replicates)),
                        seed=args.seed,
                    )
                    bootstrap_frames.append(b)
                    inventory_frames.append(inv)
            bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
            session_inventory = pd.concat(inventory_frames, ignore_index=True)
            write_df(bootstrap, out / "07_bootstrap_95ci_corrected.csv")
            write_df(session_inventory, out / "07b_session_block_inventory.csv")

            delta_frames: List[pd.DataFrame] = []
            for protocol, pairs in PAIR_SPECS.items():
                models = protocols[protocol]
                for a, b in pairs:
                    model_a, model_b = (("HYBRID", "CNN") if {a, b} == {"CNN", "HYBRID"} else (a, b))
                    print(f"[PAIRED BOOTSTRAP] {protocol} / {model_a} vs {model_b}")
                    delta_frames.append(paired_delta_ci(
                        models[model_a], models[model_b], model_a, model_b, protocol,
                        user_reps=max(100, int(args.user_replicates)),
                        session_reps=max(100, int(args.session_replicates)),
                        seed=args.seed,
                    ))
            deltas = pd.concat(delta_frames, ignore_index=True)
            write_df(deltas, out / "09_paired_performance_deltas_corrected.csv")

            signflip = paired_user_signflip_tests(protocols, args.seed)
            write_df(signflip, out / "10_exact_user_signflip_holm.csv")

            # Annotate the requested claim-level McNemar rather than discarding it.
            mc_path = source_out / "08_mcnemar_exact_holm.csv"
            mcnemar = pd.read_csv(mc_path)
            mcnemar["inference_role"] = "descriptive_sensitivity_only"
            mcnemar["dependence_caveat"] = (
                "Claims are not independent because each source window contributes multiple user claims and adjacent windows overlap; "
                "do not use claim-level McNemar alone to assert model superiority."
            )
            write_df(mcnemar, out / "08_mcnemar_exact_holm_annotated.csv")

            metrics = pd.read_csv(source_out / "05_metrics_summary.csv")
            costs = pd.read_csv(source_out / "10_cost_summary.csv")
            justification = corrected_hybrid_justification(metrics, deltas, signflip, costs)
            write_df(justification, out / "11_hybrid_cost_justification_corrected.csv")

            # Methodological notes and sanity checks.
            outside = bootstrap[~bootstrap["ci_contains_point"]]
            delta_outside = deltas[~deltas["ci_contains_point"]]
            notes = [
                {
                    "severity": "INFO",
                    "code": "CORRECTION_APPLIED",
                    "detail": "Global session resampling was replaced by stratified session-block resampling within each claimed user and class.",
                },
                {
                    "severity": "INFO",
                    "code": "CLASS_SUPPORT_PRESERVED",
                    "detail": f"All corrected session-bootstrap rows preserve genuine and impostor support in every replicate: {bool(bootstrap[bootstrap['cluster_level'].str.startswith('stratified')]['all_users_have_both_classes'].all())}.",
                },
                {
                    "severity": "INFO" if outside.empty else "WARNING",
                    "code": "BOOTSTRAP_CI_POINT_SANITY",
                    "detail": f"Corrected metric CIs excluding their original point estimate: {len(outside)} of {len(bootstrap)} rows.",
                },
                {
                    "severity": "INFO" if delta_outside.empty else "WARNING",
                    "code": "PAIRED_DELTA_CI_POINT_SANITY",
                    "detail": f"Corrected paired-delta CIs excluding their original point estimate: {len(delta_outside)} of {len(deltas)} rows.",
                },
                {
                    "severity": "LIMITATION",
                    "code": "ONE_GENUINE_TEST_SESSION_PER_USER",
                    "detail": "The test split contains one genuine source session per claimed user. Session bootstrap can estimate variation from impostor-source session composition, but cannot estimate within-user genuine-session variability.",
                },
                {
                    "severity": "LIMITATION",
                    "code": "CLAIM_LEVEL_MCNEMAR_DEPENDENCE",
                    "detail": "Claim-level McNemar is retained as requested but is anti-conservative under overlapping windows and multiple claims per source window; cluster-aware user inference is primary.",
                },
                {
                    "severity": "LIMITATION",
                    "code": "REALISTIC_BASELINES_UNAVAILABLE",
                    "detail": "Random Forest and Linear SVM remain unavailable under all-impostor evaluation because Stage 3 did not save reusable fitted models.",
                },
            ]
            write_df(pd.DataFrame(notes), out / "12_statistical_validity_notes.csv")

            # Copy stable source summaries for a self-contained corrected review.
            for name in [
                "01_protocol_availability.csv", "02_temperature_scaling.csv", "03_thresholds_per_user.csv",
                "04_overlap_and_fairness.csv", "05_metrics_summary.csv", "06_metrics_by_user.csv",
                "10_cost_summary.csv", "12_warnings_and_audit.csv", "00_pilot_manifest.json",
            ]:
                copy_if_exists(source_out / name, out / f"source_v3__{name}")

            manifest.update({
                "status": "CORRECTED_PILOT_COMPLETE_REVIEW_REQUIRED",
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(time.time() - started, 3),
                "correction": "stratified actual-session bootstrap within claimed user and class",
                "primary_inference": [
                    "claimed-user cluster bootstrap",
                    "stratified actual-session block bootstrap within claimed user and class",
                    "exact paired user sign-flip test with Holm correction",
                ],
                "claim_level_mcnemar_role": "descriptive sensitivity only",
                "bootstrap_ci_rows_excluding_point": int(len(outside)),
                "paired_delta_ci_rows_excluding_point": int(len(delta_outside)),
            })
            with (out / "00_corrected_manifest.json").open("w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

            review_zip = out / "balabit_stage5_1_pilot_v3_1_corrected_review.zip"
            with zipfile.ZipFile(review_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for p in sorted(out.iterdir()):
                    if p.is_file() and p.suffix.lower() in {".csv", ".json", ".log"}:
                        z.write(p, arcname=p.name)
            hash_path = out / "balabit_stage5_1_pilot_v3_1_corrected_review.zip.sha256.txt"
            hash_path.write_text(f"{sha256(review_zip)}  {review_zip.name}\n", encoding="utf-8")
            print("\nCorrected statistical pilot completed.")
            print("Output:", out)
            print("Review ZIP:", review_zip)
            return 0
        except Exception as exc:
            traceback.print_exc()
            manifest.update({
                "status": "FAILED_SAFE",
                "error": repr(exc),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(time.time() - started, 3),
            })
            with (out / "00_corrected_manifest.json").open("w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            print("\nFAILED SAFE. Existing Stage 3/4/5 and Pilot v3 outputs were not modified.")
            return 1
        finally:
            sys.stdout, sys.stderr = stdout0, stderr0


if __name__ == "__main__":
    raise SystemExit(main())

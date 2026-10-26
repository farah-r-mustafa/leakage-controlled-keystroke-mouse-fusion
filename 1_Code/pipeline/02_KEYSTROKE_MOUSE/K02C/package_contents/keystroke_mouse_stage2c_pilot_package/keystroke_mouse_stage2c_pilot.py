#!/usr/bin/env python3
"""Keystroke–Mouse Stage 2C
Training-only preprocessing and tensor schema freezing.

This script consumes the frozen Stage 2B review ZIP/folder. It does not create
claims and does not train authentication models.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("NumPy is required. Install with: python -m pip install numpy") from exc

VERSION = "2.0.0"
EXPECTED_STAGE2A_SPLIT_SHA256 = "e35422f0e48f96b6d1e4da99c9063379a6a488dd2bfa3e99d3e6e68857a7613d"
EXPECTED_COUNTS = {
    "keyboard_train": 15233, "keyboard_validation": 2446, "keyboard_test": 7254,
    "mouse_train": 3272, "mouse_validation": 511, "mouse_test": 751,
    "multimodal_train": 3576, "multimodal_validation": 582, "multimodal_test": 1456,
    "multimodal_primary_train": 811, "multimodal_primary_validation": 116,
    "multimodal_primary_test": 287,
}
SPLITS = ("train", "validation", "test")
LINEAGE_COLUMNS = [
    "window_id", "window_family", "window_config", "split", "user_id",
    "session_id", "segment_id", "window_order", "window_start_time",
    "window_end_time", "window_start_epoch", "window_end_epoch",
    "source_start_index", "source_end_index",
]
ROBUST_MASK_COLUMNS = [
    "keyboard_observed", "mouse_observed", "keyboard_eligible", "mouse_eligible"
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def natural_user_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", value)
    return tuple(int(x) if x.isdigit() else x.lower() for x in parts)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def locate_stage2b_root(base: Path) -> Path:
    candidates = [base]
    candidates.extend(p for p in base.rglob("stage2b_summary.json"))
    for candidate in candidates:
        root = candidate.parent if candidate.is_file() else candidate
        if (root / "stage2b_summary.json").exists() and (root / "feature_dictionary.csv").exists():
            return root
    raise FileNotFoundError("Could not locate Stage 2B root containing stage2b_summary.json")


def extract_input(input_path: Path, temp_root: Path) -> Path:
    if input_path.is_dir():
        return locate_stage2b_root(input_path)
    if not zipfile.is_zipfile(input_path):
        raise ValueError("--input-stage2b must be a Stage 2B ZIP file or extracted folder")
    with zipfile.ZipFile(input_path) as zf:
        zf.extractall(temp_root)
    return locate_stage2b_root(temp_root)


def validate_stage2b(root: Path) -> dict[str, Any]:
    required = [
        "stage2b_summary.json", "feature_dictionary.csv", "output_manifest_sha256.csv",
        "window_manifest.csv", "multimodal_primary_eligible_both_index.csv",
    ]
    for family in ("keyboard", "mouse", "multimodal"):
        for split in SPLITS:
            required += [f"{family}_features_{split}.csv", f"{family}_sequences_{split}.jsonl"]
    missing = [x for x in required if not (root / x).exists()]
    if missing:
        raise FileNotFoundError("Missing Stage 2B files: " + ", ".join(missing))

    summary = json.loads((root / "stage2b_summary.json").read_text(encoding="utf-8-sig"))
    if summary.get("status") != "PASS":
        raise ValueError("Stage 2B status is not PASS")
    split_hash = summary.get("frozen_inputs", {}).get("hashes", {}).get("frozen_session_manifest.csv")
    if split_hash != EXPECTED_STAGE2A_SPLIT_SHA256:
        raise ValueError(f"Frozen split SHA-256 mismatch: {split_hash}")

    # Verify the Stage 2B internal output manifest before consuming any artifact.
    failures = []
    for row in read_csv_rows(root / "output_manifest_sha256.csv"):
        p = root / row["relative_path"]
        if not p.exists():
            failures.append((row["relative_path"], "missing"))
        elif sha256_file(p) != row["sha256"]:
            failures.append((row["relative_path"], "hash_mismatch"))
    if failures:
        raise ValueError(f"Stage 2B internal manifest failures: {failures[:5]}")
    return summary


def load_feature_dictionary(root: Path) -> tuple[dict[str, list[str]], list[str]]:
    rows = read_csv_rows(root / "feature_dictionary.csv")
    numeric = {"keyboard": [], "mouse": []}
    masks = []
    for r in rows:
        policy = r["model_input_policy"]
        if policy == "allowed_after_training_only_preprocessing":
            fam = r["family"]
            if fam in numeric:
                numeric[fam].append(r["feature_name"])
        elif policy == "mask_allowed":
            masks.append(r["feature_name"])
    if len(numeric["keyboard"]) != 52 or len(numeric["mouse"]) != 100:
        raise ValueError(f"Unexpected feature counts: { {k: len(v) for k,v in numeric.items()} }")
    return numeric, masks


def parse_float(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        x = float(value)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def load_feature_table(path: Path, feature_columns: list[str], mask_columns: list[str] | None = None,
                       primary_only: bool = False) -> tuple[np.ndarray, list[dict[str, str]], np.ndarray]:
    mask_columns = mask_columns or []
    rows_meta: list[dict[str, str]] = []
    values: list[list[float]] = []
    robust_masks: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in feature_columns + mask_columns + LINEAGE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing_cols:
            raise ValueError(f"Missing columns in {path.name}: {missing_cols[:10]}")
        for r in reader:
            if primary_only and r.get("eligible_both") != "1":
                continue
            rows_meta.append({c: r.get(c, "") for c in LINEAGE_COLUMNS})
            values.append([parse_float(r.get(c, "")) for c in feature_columns])
            robust_masks.append([parse_float(r.get(c, "")) for c in mask_columns])
    X = np.asarray(values, dtype=np.float64)
    M = np.asarray(robust_masks, dtype=np.float64) if mask_columns else np.empty((len(values), 0), dtype=np.float64)
    return X, rows_meta, M


def fit_tabular_processor(X: np.ndarray, feature_names: list[str], variance_epsilon: float = 1e-12) -> dict[str, Any]:
    finite_counts = np.sum(np.isfinite(X), axis=0)
    all_missing = finite_counts == 0
    medians = np.full(X.shape[1], np.nan, dtype=np.float64)
    for j in range(X.shape[1]):
        vals = X[np.isfinite(X[:, j]), j]
        if vals.size:
            medians[j] = float(np.median(vals))
    Xi = X.copy()
    for j in range(X.shape[1]):
        Xi[~np.isfinite(Xi[:, j]), j] = medians[j] if math.isfinite(medians[j]) else 0.0
    means = np.mean(Xi, axis=0)
    stds = np.std(Xi, axis=0, ddof=0)
    keep = (~all_missing) & np.isfinite(stds) & (stds > variance_epsilon)
    return {
        "input_features": feature_names,
        "medians": medians,
        "means": means,
        "stds": stds,
        "keep_mask": keep,
        "all_missing_mask": all_missing,
        "variance_epsilon": variance_epsilon,
    }


def apply_tabular_processor(X: np.ndarray, proc: dict[str, Any]) -> np.ndarray:
    Xi = X.copy()
    medians = proc["medians"]
    for j in range(Xi.shape[1]):
        bad = ~np.isfinite(Xi[:, j])
        if np.any(bad):
            Xi[bad, j] = medians[j] if math.isfinite(float(medians[j])) else 0.0
    keep = proc["keep_mask"]
    Xk = Xi[:, keep]
    means = proc["means"][keep]
    stds = proc["stds"][keep]
    Z = (Xk - means) / stds
    if not np.all(np.isfinite(Z)):
        raise ValueError("Non-finite values after tabular preprocessing")
    return Z.astype(np.float32)


def processor_to_json(proc: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    keep = proc["keep_mask"]
    names = proc["input_features"]
    out = {
        "fit_scope": "training_only",
        "imputer": "median",
        "scaler": "standard_mean_std_ddof0",
        "variance_epsilon": proc["variance_epsilon"],
        "input_feature_count": len(names),
        "output_feature_count": int(np.sum(keep)),
        "kept_features": [n for n, k in zip(names, keep) if bool(k)],
        "dropped_all_missing_features": [n for n, k in zip(names, proc["all_missing_mask"]) if bool(k)],
        "dropped_zero_variance_features": [n for n, k, am in zip(names, keep, proc["all_missing_mask"]) if not bool(k) and not bool(am)],
        "parameters": [
            {
                "feature": n,
                "median": None if not math.isfinite(float(m)) else float(m),
                "mean_after_imputation": float(mu),
                "std_after_imputation": float(sd),
                "kept": bool(k),
            }
            for n, m, mu, sd, k in zip(names, proc["medians"], proc["means"], proc["stds"], keep)
        ],
    }
    if extra:
        out.update(extra)
    return out


def encode_users(metas_by_split: dict[str, list[dict[str, str]]]) -> tuple[list[str], dict[str, int]]:
    train_users = sorted({r["user_id"] for r in metas_by_split["train"]}, key=natural_user_key)
    all_users = {r["user_id"] for rows in metas_by_split.values() for r in rows}
    if set(train_users) != all_users:
        raise ValueError(f"Users outside training mapping: {sorted(all_users - set(train_users))}")
    return train_users, {u: i for i, u in enumerate(train_users)}


def save_index_csv(path: Path, metas: list[dict[str, str]], user_map: dict[str, int], extras: dict[str, list[Any]] | None = None) -> None:
    extras = extras or {}
    fields = LINEAGE_COLUMNS + ["user_index"] + list(extras.keys())
    rows = []
    for i, r in enumerate(metas):
        row = dict(r)
        row["user_index"] = user_map[r["user_id"]]
        for k, vals in extras.items():
            row[k] = vals[i]
        rows.append(row)
    write_csv(path, rows, fields)


def read_sequence_file(path: Path, family: str, primary_only: bool = False) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            o = json.loads(line)
            if primary_only and not bool(o.get("eligible_both", 0)):
                continue
            records.append(o)
    return records


def infer_step_order(records: list[dict[str, Any]], key: str) -> list[str]:
    for r in records:
        steps = r.get(key, [])
        if steps:
            return list(steps[0].keys())
    raise ValueError(f"No non-empty sequence found for {key}")


def sequence_matrix(steps: list[dict[str, Any]], order: list[str]) -> np.ndarray:
    arr = np.empty((len(steps), len(order)), dtype=np.float64)
    for i, step in enumerate(steps):
        for j, name in enumerate(order):
            v = step.get(name)
            try:
                x = float(v)
                arr[i, j] = x if math.isfinite(x) else np.nan
            except Exception:
                arr[i, j] = np.nan
    return arr


def fit_sequence_processor(records: list[dict[str, Any]], key: str, order: list[str], epsilon: float = 1e-12) -> dict[str, Any]:
    chunks = [sequence_matrix(r.get(key, []), order) for r in records if r.get(key)]
    if not chunks:
        raise ValueError(f"No steps available to fit {key}")
    X = np.vstack(chunks)
    medians = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        vals = X[np.isfinite(X[:, j]), j]
        if vals.size:
            medians[j] = np.median(vals)
        X[~np.isfinite(X[:, j]), j] = medians[j] if math.isfinite(float(medians[j])) else 0.0
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    keep = np.isfinite(stds) & (stds > epsilon)
    return {
        "input_features": order, "medians": medians, "means": means, "stds": stds,
        "keep_mask": keep, "variance_epsilon": epsilon, "training_valid_steps": int(X.shape[0])
    }


def apply_sequence_processor(steps: list[dict[str, Any]], proc: dict[str, Any]) -> np.ndarray:
    X = sequence_matrix(steps, proc["input_features"])
    for j in range(X.shape[1]):
        bad = ~np.isfinite(X[:, j])
        if np.any(bad):
            med = proc["medians"][j]
            X[bad, j] = med if math.isfinite(float(med)) else 0.0
    keep = proc["keep_mask"]
    X = (X[:, keep] - proc["means"][keep]) / proc["stds"][keep]
    if not np.all(np.isfinite(X)):
        raise ValueError("Non-finite sequence tensor after preprocessing")
    return X.astype(np.float32)


def round_up(value: float, multiple: int) -> int:
    return max(multiple, int(math.ceil(value / multiple) * multiple))


def training_length_policy(records: list[dict[str, Any]], key: str, mode: str) -> dict[str, Any]:
    lengths = np.asarray([len(r.get(key, [])) for r in records], dtype=np.int32)
    positive = lengths[lengths > 0]
    if positive.size == 0:
        raise ValueError(f"No positive lengths for {key}")
    if mode == "primary":
        # Preserve every training sequence, rounded to a compact multiple.
        multiple = 8
        max_len = round_up(float(positive.max()), multiple)
        rule = "training_max_rounded_up_to_multiple_of_8"
        quantile = None
    else:
        multiple = 8
        q = float(np.quantile(positive, 0.995))
        max_len = round_up(q, multiple)
        rule = "training_positive_length_q99.5_rounded_up_to_multiple_of_8"
        quantile = q
    return {
        "max_length": int(max_len), "rule": rule, "training_max": int(positive.max()),
        "training_median": float(np.median(positive)), "training_q95": float(np.quantile(positive, 0.95)),
        "training_q99": float(np.quantile(positive, 0.99)), "training_q995": quantile,
        "zero_length_records": int(np.sum(lengths == 0)), "positive_records": int(positive.size),
    }


def pad_or_downsample(X: np.ndarray, max_len: int) -> tuple[np.ndarray, np.ndarray, bool]:
    out = np.zeros((max_len, X.shape[1]), dtype=np.float32)
    mask = np.zeros(max_len, dtype=np.uint8)
    n = X.shape[0]
    if n == 0:
        return out, mask, False
    truncated = n > max_len
    if truncated:
        # Deterministic uniform temporal selection preserves the complete 10-second span.
        idx = np.linspace(0, n - 1, num=max_len, dtype=np.int64)
        selected = X[idx]
        used = max_len
    else:
        selected = X
        used = n
    out[:used] = selected
    mask[:used] = 1
    return out, mask, truncated


def sequence_processor_json(proc: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    keep = proc["keep_mask"]
    return {
        "fit_scope": "training_only_valid_steps",
        "imputer": "per_channel_median",
        "scaler": "per_channel_standard_mean_std_ddof0",
        "padding_value_after_scaling": 0.0,
        "padding_side": "right",
        "overlength_policy": "deterministic_uniform_temporal_downsampling_preserve_endpoints",
        "input_channel_count": len(proc["input_features"]),
        "output_channel_count": int(np.sum(keep)),
        "training_valid_steps": proc["training_valid_steps"],
        "kept_channels": [n for n,k in zip(proc["input_features"], keep) if bool(k)],
        "dropped_zero_variance_channels": [n for n,k in zip(proc["input_features"], keep) if not bool(k)],
        "length_policy": policy,
        "parameters": [
            {"channel": n, "median": float(m), "mean": float(mu), "std": float(sd), "kept": bool(k)}
            for n,m,mu,sd,k in zip(proc["input_features"], proc["medians"], proc["means"], proc["stds"], keep)
        ],
    }


def build_sequence_npz(records: list[dict[str, Any]], family: str, proc_a: dict[str, Any], max_a: int,
                       user_map: dict[str, int], proc_b: dict[str, Any] | None = None,
                       max_b: int | None = None) -> tuple[dict[str, np.ndarray], list[dict[str, str]], list[dict[str, Any]]]:
    metas = []
    audit = []
    if family in ("keyboard", "mouse"):
        key = "steps"
        Xs=[]; masks=[]; ys=[]; trunc=[]; lengths=[]
        for r in records:
            x = apply_sequence_processor(r[key], proc_a)
            padded, mask, was_trunc = pad_or_downsample(x, max_a)
            Xs.append(padded); masks.append(mask); ys.append(user_map[r["user_id"]])
            trunc.append(was_trunc); lengths.append(len(r[key]))
            metas.append({c: str(r.get(c, "")) for c in ["window_id","split","user_id","session_id","segment_id"]})
            audit.append({"window_id":r["window_id"],"branch":family,"original_length":len(r[key]),"max_length":max_a,"was_downsampled":int(was_trunc)})
        arrays={"X":np.stack(Xs),"mask":np.stack(masks),"y_user":np.asarray(ys,dtype=np.int16),
                "original_length":np.asarray(lengths,dtype=np.int16),"was_downsampled":np.asarray(trunc,dtype=np.uint8)}
        return arrays, metas, audit

    assert proc_b is not None and max_b is not None
    kx=[]; km=[]; mx=[]; mm=[]; ys=[]; klen=[]; mlen=[]; ktr=[]; mtr=[]; modmask=[]; primary=[]
    for r in records:
        k = apply_sequence_processor(r.get("keyboard_steps",[]), proc_a)
        m = apply_sequence_processor(r.get("mouse_steps",[]), proc_b)
        kp,k_mask,k_t = pad_or_downsample(k,max_a)
        mp,m_mask,m_t = pad_or_downsample(m,max_b)
        kx.append(kp); km.append(k_mask); mx.append(mp); mm.append(m_mask)
        ys.append(user_map[r["user_id"]]); klen.append(len(r.get("keyboard_steps",[]))); mlen.append(len(r.get("mouse_steps",[])))
        ktr.append(k_t); mtr.append(m_t)
        modmask.append([int(len(r.get("keyboard_steps",[]))>0),int(len(r.get("mouse_steps",[]))>0)])
        primary.append(int(bool(r.get("eligible_both",0))))
        metas.append({c: str(r.get(c,"")) for c in ["window_id","split","user_id","session_id","segment_id"]})
        audit += [
            {"window_id":r["window_id"],"branch":"multimodal_keyboard","original_length":len(r.get("keyboard_steps",[])),"max_length":max_a,"was_downsampled":int(k_t)},
            {"window_id":r["window_id"],"branch":"multimodal_mouse","original_length":len(r.get("mouse_steps",[])),"max_length":max_b,"was_downsampled":int(m_t)},
        ]
    arrays={"keyboard_X":np.stack(kx),"keyboard_mask":np.stack(km),"mouse_X":np.stack(mx),"mouse_mask":np.stack(mm),
            "modality_mask":np.asarray(modmask,dtype=np.uint8),"y_user":np.asarray(ys,dtype=np.int16),
            "keyboard_original_length":np.asarray(klen,dtype=np.int16),"mouse_original_length":np.asarray(mlen,dtype=np.int16),
            "keyboard_was_downsampled":np.asarray(ktr,dtype=np.uint8),"mouse_was_downsampled":np.asarray(mtr,dtype=np.uint8),
            "primary_multimodal":np.asarray(primary,dtype=np.uint8)}
    return arrays, metas, audit


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def zip_output(output_dir: Path) -> Path:
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(output_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=f"{output_dir.name}/{p.relative_to(output_dir).as_posix()}")
    return zip_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2C training-only preprocessing and tensor schema freezing")
    ap.add_argument("--input-stage2b", required=True, help="Stage 2B review ZIP or extracted folder")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    started = time.time(); started_iso = now_iso()
    input_path = Path(args.input_stage2b).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Use --overwrite.")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    warnings: list[dict[str, Any]]=[]; failures: list[dict[str, Any]]=[]
    log=[f"Stage 2C v{VERSION}",f"Started: {started_iso}",f"Input: {input_path}",f"Output: {output}"]

    try:
        with tempfile.TemporaryDirectory(prefix="km_stage2c_") as td:
            root=extract_input(input_path,Path(td))
            summary2b=validate_stage2b(root)
            log.append(f"Validated Stage 2B root: {root}")
            numeric,masks=load_feature_dictionary(root)
            feature_sets={
                "keyboard":numeric["keyboard"],
                "mouse":numeric["mouse"],
                "multimodal_primary":numeric["keyboard"]+numeric["mouse"],
                "multimodal_any":numeric["keyboard"]+numeric["mouse"]+ROBUST_MASK_COLUMNS,
            }

            user_mapping_global=None
            tabular_summary=[]; feature_order_rows=[]; preprocessing_audit=[]
            for cohort in ("keyboard","mouse","multimodal_primary","multimodal_any"):
                source_family = "multimodal" if cohort.startswith("multimodal") else cohort
                primary_only = cohort=="multimodal_primary"
                base_features = numeric["keyboard"]+numeric["mouse"] if source_family=="multimodal" else numeric[source_family]
                mask_cols = ROBUST_MASK_COLUMNS if cohort=="multimodal_any" else []
                raw={}; metas={}; mask_arrays={}
                for split in SPLITS:
                    raw[split],metas[split],mask_arrays[split]=load_feature_table(
                        root/f"{source_family}_features_{split}.csv",base_features,mask_cols,primary_only)
                    expected_key=f"{cohort}_{split}" if cohort=="multimodal_primary" else f"{source_family}_{split}"
                    expected=EXPECTED_COUNTS.get(expected_key)
                    if expected is not None and len(metas[split])!=expected:
                        raise ValueError(f"Count mismatch {cohort}/{split}: {len(metas[split])} != {expected}")
                if mask_cols:
                    raw_for_fit=np.hstack([raw["train"],mask_arrays["train"]])
                    names=base_features+mask_cols
                else:
                    raw_for_fit=raw["train"]; names=base_features
                proc=fit_tabular_processor(raw_for_fit,names)
                users,user_map=encode_users(metas)
                if user_mapping_global is None:
                    user_mapping_global=(users,user_map)
                elif users!=user_mapping_global[0]:
                    raise ValueError(f"User mapping differs for cohort {cohort}")
                proc_json=processor_to_json(proc,{"cohort":cohort,"mask_features_included":mask_cols})
                (output/"processors").mkdir(exist_ok=True)
                (output/"processors"/f"tabular_{cohort}_processor.json").write_text(json.dumps(proc_json,indent=2),encoding="utf-8")
                for pos,n in enumerate(proc_json["kept_features"]):
                    feature_order_rows.append({"cohort":cohort,"position":pos,"feature_name":n})
                for split in SPLITS:
                    X0=np.hstack([raw[split],mask_arrays[split]]) if mask_cols else raw[split]
                    X=apply_tabular_processor(X0,proc)
                    y=np.asarray([user_map[r["user_id"]] for r in metas[split]],dtype=np.int16)
                    save_npz(output/"tabular"/f"{cohort}_{split}.npz",{"X":X,"y_user":y})
                    save_index_csv(output/"indexes"/f"{cohort}_{split}_index.csv",metas[split],user_map)
                    preprocessing_audit.append({
                        "representation":"tabular","cohort":cohort,"split":split,"records":len(metas[split]),
                        "input_features":len(names),"output_features":X.shape[1],"nonfinite_after":int(np.size(X)-np.isfinite(X).sum()),
                        "fit_scope":"training_only" if split=="train" else "transform_only_using_training_parameters"
                    })
                tabular_summary.append({"cohort":cohort,"input_features":len(names),"output_features":len(proc_json["kept_features"]),
                    "dropped_all_missing":len(proc_json["dropped_all_missing_features"]),"dropped_zero_variance":len(proc_json["dropped_zero_variance_features"])})

            users,user_map=user_mapping_global
            write_csv(output/"user_label_mapping.csv",[{"user_id":u,"user_index":user_map[u]} for u in users],["user_id","user_index"])
            write_csv(output/"frozen_feature_order.csv",feature_order_rows,["cohort","position","feature_name"])

            # Sequence records and training-only processors.
            seq_records: dict[str, dict[str,list[dict[str,Any]]]]={}
            for family in ("keyboard","mouse"):
                seq_records[family]={split:read_sequence_file(root/f"{family}_sequences_{split}.jsonl",family) for split in SPLITS}
            seq_records["multimodal_any"]={split:read_sequence_file(root/f"multimodal_sequences_{split}.jsonl","multimodal") for split in SPLITS}
            seq_records["multimodal_primary"]={split:read_sequence_file(root/f"multimodal_sequences_{split}.jsonl","multimodal",True) for split in SPLITS}

            k_order=infer_step_order(seq_records["keyboard"]["train"],"steps")
            m_order=infer_step_order(seq_records["mouse"]["train"],"steps")
            mmk_order=infer_step_order(seq_records["multimodal_any"]["train"],"keyboard_steps")
            mmm_order=infer_step_order(seq_records["multimodal_any"]["train"],"mouse_steps")
            if k_order!=mmk_order or m_order!=mmm_order:
                raise ValueError("Sequence channel order mismatch between baseline and multimodal files")

            # Fixed baseline processors.
            k_proc=fit_sequence_processor(seq_records["keyboard"]["train"],"steps",k_order)
            m_proc=fit_sequence_processor(seq_records["mouse"]["train"],"steps",m_order)
            k_policy={"max_length":5,"rule":"frozen_fixed_keyboard_window","training_max":5}
            m_policy={"max_length":10,"rule":"frozen_fixed_mouse_window","training_max":10}

            # Primary and robust multimodal get independent training-only preprocessors.
            mm_processors={}
            for cohort,mode in (("multimodal_primary","primary"),("multimodal_any","robust")):
                rec=seq_records[cohort]["train"]
                kp=fit_sequence_processor(rec,"keyboard_steps",k_order)
                mp=fit_sequence_processor(rec,"mouse_steps",m_order)
                kl=training_length_policy(rec,"keyboard_steps",mode)
                ml=training_length_policy(rec,"mouse_steps",mode)
                mm_processors[cohort]=(kp,mp,kl,ml)

            seq_proc_specs=[("keyboard",k_proc,k_policy),("mouse",m_proc,m_policy)]
            for name,proc,policy in seq_proc_specs:
                (output/"processors"/f"sequence_{name}_processor.json").write_text(json.dumps(sequence_processor_json(proc,policy),indent=2),encoding="utf-8")
            for cohort,(kp,mp,kl,ml) in mm_processors.items():
                (output/"processors"/f"sequence_{cohort}_keyboard_processor.json").write_text(json.dumps(sequence_processor_json(kp,kl),indent=2),encoding="utf-8")
                (output/"processors"/f"sequence_{cohort}_mouse_processor.json").write_text(json.dumps(sequence_processor_json(mp,ml),indent=2),encoding="utf-8")

            sequence_audit=[]; tensor_rows=[]
            for family,proc,policy in seq_proc_specs:
                for split in SPLITS:
                    arrays,metas,audit=build_sequence_npz(seq_records[family][split],family,proc,int(policy["max_length"]),user_map)
                    save_npz(output/"sequences"/f"{family}_{split}.npz",arrays)
                    save_index_csv(output/"indexes"/f"{family}_sequence_{split}_index.csv",metas,user_map)
                    sequence_audit.extend({**a,"cohort":family,"split":split} for a in audit)
                    tensor_rows.append({"cohort":family,"split":split,"records":len(metas),"tensor_a_shape":str(arrays["X"].shape),"tensor_b_shape":"","downsampled_a":int(arrays["was_downsampled"].sum()),"downsampled_b":0})
                    preprocessing_audit.append({"representation":"sequence","cohort":family,"split":split,"records":len(metas),"input_features":len(proc["input_features"]),"output_features":int(proc["keep_mask"].sum()),"nonfinite_after":int(np.size(arrays["X"])-np.isfinite(arrays["X"]).sum()),"fit_scope":"training_only" if split=="train" else "transform_only_using_training_parameters"})

            for cohort,(kp,mp,kl,ml) in mm_processors.items():
                for split in SPLITS:
                    arrays,metas,audit=build_sequence_npz(seq_records[cohort][split],"multimodal",kp,int(kl["max_length"]),user_map,mp,int(ml["max_length"]))
                    save_npz(output/"sequences"/f"{cohort}_{split}.npz",arrays)
                    save_index_csv(output/"indexes"/f"{cohort}_sequence_{split}_index.csv",metas,user_map,
                                   {"primary_multimodal":arrays["primary_multimodal"].tolist()})
                    sequence_audit.extend({**a,"cohort":cohort,"split":split} for a in audit)
                    tensor_rows.append({"cohort":cohort,"split":split,"records":len(metas),"tensor_a_shape":str(arrays["keyboard_X"].shape),"tensor_b_shape":str(arrays["mouse_X"].shape),"downsampled_a":int(arrays["keyboard_was_downsampled"].sum()),"downsampled_b":int(arrays["mouse_was_downsampled"].sum())})
                    preprocessing_audit.append({"representation":"sequence","cohort":cohort,"split":split,"records":len(metas),"input_features":len(kp["input_features"])+len(mp["input_features"]),"output_features":int(kp["keep_mask"].sum()+mp["keep_mask"].sum()),"nonfinite_after":int(np.size(arrays["keyboard_X"])-np.isfinite(arrays["keyboard_X"]).sum()+np.size(arrays["mouse_X"])-np.isfinite(arrays["mouse_X"]).sum()),"fit_scope":"training_only" if split=="train" else "transform_only_using_training_parameters"})

            write_csv(output/"tabular_preprocessing_summary.csv",tabular_summary,["cohort","input_features","output_features","dropped_all_missing","dropped_zero_variance"])
            write_csv(output/"preprocessing_audit.csv",preprocessing_audit,["representation","cohort","split","records","input_features","output_features","nonfinite_after","fit_scope"])
            write_csv(output/"tensor_shapes_and_truncation.csv",tensor_rows,["cohort","split","records","tensor_a_shape","tensor_b_shape","downsampled_a","downsampled_b"])
            write_csv(output/"sequence_window_audit.csv",sequence_audit,["cohort","split","window_id","branch","original_length","max_length","was_downsampled"])

            # Leakage/integrity checks using all output indexes.
            leakage=[]
            for cohort in ("keyboard","mouse","multimodal_primary","multimodal_any"):
                session_splits=defaultdict(set); segment_splits=defaultdict(set); window_seen=set(); dup=0
                for split in SPLITS:
                    rows=read_csv_rows(output/"indexes"/f"{cohort}_{split}_index.csv")
                    for r in rows:
                        if r["window_id"] in window_seen: dup+=1
                        window_seen.add(r["window_id"]); session_splits[r["session_id"]].add(split); segment_splits[r["segment_id"]].add(split)
                leakage.append({"cohort":cohort,"duplicate_window_ids":dup,"session_split_conflicts":sum(len(v)>1 for v in session_splits.values()),"segment_split_conflicts":sum(len(v)>1 for v in segment_splits.values()),"status":"PASS" if dup==0 and all(len(v)==1 for v in session_splits.values()) and all(len(v)==1 for v in segment_splits.values()) else "FAIL"})
            write_csv(output/"leakage_integrity_checks.csv",leakage,["cohort","duplicate_window_ids","session_split_conflicts","segment_split_conflicts","status"])
            if any(r["status"]!="PASS" for r in leakage):
                raise ValueError("Leakage integrity failure")

            # Frozen tensor schema.
            tensor_schema={
                "stage":"Keystroke–Mouse Stage 2C","version":VERSION,"fit_policy":"all fitted parameters use Training only",
                "user_mapping":"user_label_mapping.csv",
                "tabular":{
                    "dtype":"float32","labels":"int16 user_index","files":"tabular/<cohort>_<split>.npz",
                    "cohorts":["keyboard","mouse","multimodal_primary","multimodal_any"],
                    "multimodal_primary_scope":"eligible_both only",
                    "multimodal_any_scope":"eligible_any with four non-redundant modality masks",
                },
                "sequences":{
                    "dtype":"float32","mask_dtype":"uint8","padding":"right zero after standardization",
                    "keyboard":{"length":5},"mouse":{"length":10},
                    "multimodal_primary":{"keyboard_length":mm_processors["multimodal_primary"][2]["max_length"],"mouse_length":mm_processors["multimodal_primary"][3]["max_length"]},
                    "multimodal_any":{"keyboard_length":mm_processors["multimodal_any"][2]["max_length"],"mouse_length":mm_processors["multimodal_any"][3]["max_length"]},
                },
                "prohibited_actions":["no claim construction","no model training","no validation/test fitting","no threshold selection","no feature selection using validation/test"],
            }
            (output/"tensor_schema.json").write_text(json.dumps(tensor_schema,indent=2),encoding="utf-8")

            # Human-readable decision report.
            report=f"""# Stage 2C — Frozen preprocessing decisions\n\n- Source Stage 2B status: PASS.\n- Frozen Stage 2A split SHA-256: `{EXPECTED_STAGE2A_SPLIT_SHA256}`.\n- Imputation: per-feature/per-channel median fitted on Training only.\n- Scaling: mean/std (ddof=0) fitted on Training only.\n- All-missing and zero-variance fields are determined from Training only.\n- No clipping, winsorization, feature selection, claims, calibration, thresholds, or model training.\n- Keyboard baseline sequence: 5 steps.\n- Mouse baseline sequence: 10 steps.\n- Multimodal primary lengths: keyboard {mm_processors['multimodal_primary'][2]['max_length']}, mouse {mm_processors['multimodal_primary'][3]['max_length']}.\n- Multimodal eligible-any lengths: keyboard {mm_processors['multimodal_any'][2]['max_length']}, mouse {mm_processors['multimodal_any'][3]['max_length']}.\n- Overlength sequences use deterministic uniform temporal downsampling preserving endpoints.\n- Short sequences are right-padded with zero after standardization and accompanied by masks.\n- `multimodal_primary` includes eligible-both windows only.\n- `multimodal_any` retains missing-modality windows and only four non-redundant modality masks.\n"""
            (output/"stage2c_frozen_decisions.md").write_text(report,encoding="utf-8")

            finished_iso=now_iso(); elapsed=time.time()-started
            summary={
                "stage":"Keystroke–Mouse Stage 2C","version":VERSION,"status":"PASS","started_at":started_iso,"finished_at":finished_iso,"elapsed_seconds":round(elapsed,3),
                "input_stage2b":{"path":str(input_path),"sha256":sha256_file(input_path) if input_path.is_file() else None,"stage2a_split_sha256":EXPECTED_STAGE2A_SPLIT_SHA256},
                "users":users,"tabular":tabular_summary,"tensor_shapes":tensor_rows,
                "warning_count":len(warnings),"failure_count":0,
            }
            (output/"stage2c_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
            log += [f"Finished: {finished_iso}",f"Elapsed seconds: {elapsed:.3f}","Status: PASS"]
    except Exception as exc:
        failures.append({"severity":"ERROR","code":type(exc).__name__,"message":str(exc)})
        log += [f"FAILED: {type(exc).__name__}: {exc}",f"Finished: {now_iso()}","Status: FAIL"]
        write_csv(output/"warnings_and_failures.csv",warnings+failures,["severity","code","message"])
        (output/"execution_log.txt").write_text("\n".join(log)+"\n",encoding="utf-8")
        raise

    write_csv(output/"warnings_and_failures.csv",warnings+failures,["severity","code","message"])
    (output/"execution_log.txt").write_text("\n".join(log)+"\n",encoding="utf-8")

    # Manifest excludes itself to avoid self-reference.
    manifest=[]
    for p in sorted(output.rglob("*")):
        if p.is_file() and p.name!="output_manifest_sha256.csv":
            manifest.append({"relative_path":p.relative_to(output).as_posix(),"size_bytes":p.stat().st_size,"sha256":sha256_file(p)})
    write_csv(output/"output_manifest_sha256.csv",manifest,["relative_path","size_bytes","sha256"])
    zip_path=zip_output(output)
    print(f"PASS: {output}")
    print(f"ZIP:  {zip_path}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

#!/usr/bin/env python3
"""Keystroke–Mouse Stage 3A
Claim construction and authentication protocol freezing.

Consumes the frozen Stage 2C review ZIP/folder. It creates deterministic,
model-agnostic Genuine/Impostor claim tables for:
  1) balanced_1to1
  2) realistic_all_impostor

No model training, score calibration, threshold selection, or test-driven tuning
is performed in this stage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("NumPy is required. Install with: python -m pip install numpy") from exc

VERSION = "3.0.0"
BASE_SEED = 31082026
EXPECTED_STAGE2A_SPLIT_SHA256 = "e35422f0e48f96b6d1e4da99c9063379a6a488dd2bfa3e99d3e6e68857a7613d"
COHORTS = ("keyboard", "mouse", "multimodal_primary", "multimodal_any")
SPLITS = ("train", "validation", "test")
PROTOCOLS = ("balanced_1to1", "realistic_all_impostor")
EXPECTED_COUNTS = {
    "keyboard": {"train": 15233, "validation": 2446, "test": 7254},
    "mouse": {"train": 3272, "validation": 511, "test": 751},
    "multimodal_primary": {"train": 811, "validation": 116, "test": 287},
    "multimodal_any": {"train": 3576, "validation": 582, "test": 1456},
}
CLAIM_FIELDS = [
    "claim_id", "cohort", "protocol", "split", "source_row_index",
    "window_id", "session_id", "segment_id", "window_start_epoch",
    "window_end_epoch", "actual_user_id", "actual_user_index",
    "claimed_user_id", "claimed_user_index", "is_genuine",
    "sampling_role", "sampling_seed"
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def natural_user_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(x) if x.isdigit() else x.lower() for x in parts)


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_digest(*parts: Any) -> str:
    text = "|".join(str(x) for x in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_rank(*parts: Any) -> int:
    return int(stable_digest(*parts)[:16], 16)


def deterministic_claim_id(cohort: str, protocol: str, split: str,
                           window_id: str, claimed_user_id: str) -> str:
    return "CL_" + stable_digest("stage3a", VERSION, cohort, protocol, split,
                                  window_id, claimed_user_id)[:24]


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


def locate_stage2c_root(base: Path) -> Path:
    candidates = [base]
    candidates.extend(p.parent for p in base.rglob("stage2c_summary.json"))
    for root in candidates:
        if ((root / "stage2c_summary.json").exists()
                and (root / "indexes").is_dir()
                and (root / "tabular").is_dir()):
            return root
    raise FileNotFoundError("Could not locate Stage 2C root containing stage2c_summary.json")


def extract_input(input_path: Path, temp_root: Path) -> Path:
    if input_path.is_dir():
        return locate_stage2c_root(input_path)
    if not zipfile.is_zipfile(input_path):
        raise ValueError("--input-stage2c must be a Stage 2C ZIP file or extracted folder")
    with zipfile.ZipFile(input_path) as zf:
        zf.extractall(temp_root)
    return locate_stage2c_root(temp_root)


def verify_internal_manifest(root: Path) -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    manifest = root / "output_manifest_sha256.csv"
    if not manifest.exists():
        return [("output_manifest_sha256.csv", "missing")]
    for row in read_csv_rows(manifest):
        rel = row.get("relative_path", "")
        p = root / rel
        if not p.exists():
            failures.append((rel, "missing"))
        elif sha256_file(p) != row.get("sha256"):
            failures.append((rel, "hash_mismatch"))
    return failures


def validate_stage2c(root: Path) -> tuple[dict[str, Any], list[str], dict[str, int]]:
    required = [
        "stage2c_summary.json", "output_manifest_sha256.csv",
        "user_label_mapping.csv", "leakage_integrity_checks.csv",
    ]
    for cohort in COHORTS:
        for split in SPLITS:
            required += [
                f"indexes/{cohort}_{split}_index.csv",
                f"tabular/{cohort}_{split}.npz",
            ]
    missing = [x for x in required if not (root / x).exists()]
    if missing:
        raise FileNotFoundError("Missing Stage 2C files: " + ", ".join(missing[:12]))

    summary = json.loads((root / "stage2c_summary.json").read_text(encoding="utf-8-sig"))
    if summary.get("status") != "PASS":
        raise ValueError("Stage 2C status is not PASS")
    split_hash = summary.get("input_stage2b", {}).get("stage2a_split_sha256")
    if split_hash != EXPECTED_STAGE2A_SPLIT_SHA256:
        raise ValueError(f"Frozen Stage 2A split SHA-256 mismatch: {split_hash}")

    failures = verify_internal_manifest(root)
    if failures:
        raise ValueError(f"Stage 2C internal manifest failures: {failures[:5]}")

    mapping_rows = read_csv_rows(root / "user_label_mapping.csv")
    mapping_rows.sort(key=lambda r: int(r["user_index"]))
    users = [r["user_id"] for r in mapping_rows]
    user_to_idx = {r["user_id"]: int(r["user_index"]) for r in mapping_rows}
    if users != [f"User{i}" for i in range(1, 12)]:
        raise ValueError(f"Unexpected frozen users: {users}")
    return summary, users, user_to_idx


def load_index_and_validate(root: Path, cohort: str, split: str,
                            user_to_idx: dict[str, int]) -> list[dict[str, Any]]:
    index_path = root / "indexes" / f"{cohort}_{split}_index.csv"
    rows = read_csv_rows(index_path)
    expected = EXPECTED_COUNTS[cohort][split]
    if len(rows) != expected:
        raise ValueError(f"Unexpected {cohort}/{split} index rows: {len(rows)} != {expected}")

    z = np.load(root / "tabular" / f"{cohort}_{split}.npz", allow_pickle=False)
    if "y_user" not in z.files:
        raise ValueError(f"Missing y_user in {cohort}_{split}.npz")
    y = z["y_user"]
    if len(y) != len(rows):
        raise ValueError(f"Index/NPZ row mismatch for {cohort}/{split}")

    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        uid = r.get("user_id", "")
        wid = r.get("window_id", "")
        if uid not in user_to_idx:
            raise ValueError(f"Unknown user {uid} in {index_path.name}")
        idx = int(r.get("user_index", -1))
        if idx != user_to_idx[uid] or int(y[i]) != idx:
            raise ValueError(f"User label mismatch at {cohort}/{split} row {i}")
        if not wid or wid in seen:
            raise ValueError(f"Duplicate/empty window_id at {cohort}/{split} row {i}: {wid}")
        seen.add(wid)
        result.append({
            "source_row_index": i,
            "window_id": wid,
            "session_id": r.get("session_id", ""),
            "segment_id": r.get("segment_id", ""),
            "window_start_epoch": r.get("window_start_epoch", ""),
            "window_end_epoch": r.get("window_end_epoch", ""),
            "actual_user_id": uid,
            "actual_user_index": idx,
        })
    return result


def allocate_balanced_quotas(target: int, capacities: dict[str, int],
                             cohort: str, split: str, claimed_user: str) -> dict[str, int]:
    if target < 0:
        raise ValueError("Negative target")
    total_capacity = sum(capacities.values())
    if target > total_capacity:
        raise ValueError(
            f"Not enough unique impostor windows for {cohort}/{split}/{claimed_user}: "
            f"target={target}, capacity={total_capacity}"
        )
    order = sorted(
        capacities,
        key=lambda u: stable_rank(BASE_SEED, cohort, split, claimed_user, "user_order", u)
    )
    quotas = {u: 0 for u in capacities}
    remaining = target
    while remaining > 0:
        progressed = False
        for u in order:
            if quotas[u] < capacities[u]:
                quotas[u] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Quota allocation stalled unexpectedly")
    return quotas


def session_aware_select(candidates: list[dict[str, Any]], quota: int,
                         cohort: str, split: str, claimed_user: str,
                         actual_user: str) -> list[dict[str, Any]]:
    if quota == 0:
        return []
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_session[str(row["session_id"])].append(row)
    session_order = sorted(
        by_session,
        key=lambda s: stable_rank(BASE_SEED, cohort, split, claimed_user,
                                  actual_user, "session_order", s)
    )
    for s in session_order:
        by_session[s].sort(
            key=lambda r: stable_rank(BASE_SEED, cohort, split, claimed_user,
                                      actual_user, s, "window_order", r["window_id"])
        )
    positions = {s: 0 for s in session_order}
    selected: list[dict[str, Any]] = []
    while len(selected) < quota:
        progressed = False
        for s in session_order:
            pos = positions[s]
            if pos < len(by_session[s]):
                selected.append(by_session[s][pos])
                positions[s] += 1
                progressed = True
                if len(selected) == quota:
                    break
        if not progressed:
            raise RuntimeError("Session-aware selection exhausted before quota")
    return selected


def make_claim(row: dict[str, Any], cohort: str, protocol: str, split: str,
               claimed_user: str, claimed_index: int, role: str) -> dict[str, Any]:
    genuine = int(row["actual_user_id"] == claimed_user)
    return {
        "claim_id": deterministic_claim_id(cohort, protocol, split, row["window_id"], claimed_user),
        "cohort": cohort,
        "protocol": protocol,
        "split": split,
        "source_row_index": row["source_row_index"],
        "window_id": row["window_id"],
        "session_id": row["session_id"],
        "segment_id": row["segment_id"],
        "window_start_epoch": row["window_start_epoch"],
        "window_end_epoch": row["window_end_epoch"],
        "actual_user_id": row["actual_user_id"],
        "actual_user_index": row["actual_user_index"],
        "claimed_user_id": claimed_user,
        "claimed_user_index": claimed_index,
        "is_genuine": genuine,
        "sampling_role": role,
        "sampling_seed": BASE_SEED,
    }


def build_balanced_claims(rows: list[dict[str, Any]], cohort: str, split: str,
                          users: list[str], user_to_idx: dict[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_user[row["actual_user_id"]].append(row)
    claims: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for claimed in users:
        genuine_rows = sorted(by_user[claimed], key=lambda r: int(r["source_row_index"]))
        target = len(genuine_rows)
        capacities = {u: len(by_user[u]) for u in users if u != claimed}
        quotas = allocate_balanced_quotas(target, capacities, cohort, split, claimed)
        selected: list[dict[str, Any]] = []
        for actual in users:
            if actual == claimed:
                continue
            chosen = session_aware_select(by_user[actual], quotas[actual], cohort, split, claimed, actual)
            selected.extend(chosen)
            audit.append({
                "cohort": cohort,
                "split": split,
                "claimed_user_id": claimed,
                "actual_impostor_user_id": actual,
                "available_windows": capacities[actual],
                "selected_windows": quotas[actual],
                "available_sessions": len({str(r["session_id"]) for r in by_user[actual]}),
                "selected_sessions": len({str(r["session_id"]) for r in chosen}),
                "sampling_seed": BASE_SEED,
            })
        selected.sort(key=lambda r: (
            user_to_idx[r["actual_user_id"]],
            stable_rank(BASE_SEED, cohort, split, claimed, "final", r["window_id"])
        ))
        claims.extend(make_claim(r, cohort, "balanced_1to1", split,
                                 claimed, user_to_idx[claimed], "genuine_all")
                      for r in genuine_rows)
        claims.extend(make_claim(r, cohort, "balanced_1to1", split,
                                 claimed, user_to_idx[claimed], "sampled_impostor")
                      for r in selected)
    return claims, audit


def build_all_impostor_claims(rows: list[dict[str, Any]], cohort: str, split: str,
                              users: list[str], user_to_idx: dict[str, int]) -> list[dict[str, Any]]:
    ordered_rows = sorted(rows, key=lambda r: int(r["source_row_index"]))
    claims: list[dict[str, Any]] = []
    for claimed in users:
        for row in ordered_rows:
            role = "genuine_all" if row["actual_user_id"] == claimed else "all_impostor"
            claims.append(make_claim(row, cohort, "realistic_all_impostor", split,
                                     claimed, user_to_idx[claimed], role))
    return claims


def inspect_claims(claims: list[dict[str, Any]], source_rows: list[dict[str, Any]],
                   cohort: str, protocol: str, split: str, users: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    counts: list[dict[str, Any]] = []
    source_ids = {r["window_id"] for r in source_rows}
    source_map = {r["window_id"]: r for r in source_rows}
    claim_ids = [r["claim_id"] for r in claims]
    pairs = [(r["window_id"], r["claimed_user_id"]) for r in claims]

    def add_check(name: str, observed: int, expected: int = 0) -> None:
        checks.append({
            "cohort": cohort, "protocol": protocol, "split": split,
            "check": name, "observed": observed, "expected": expected,
            "status": "PASS" if observed == expected else "FAIL",
        })

    add_check("duplicate_claim_ids", len(claim_ids) - len(set(claim_ids)))
    add_check("duplicate_window_claimed_pairs", len(pairs) - len(set(pairs)))
    add_check("unknown_source_windows", sum(r["window_id"] not in source_ids for r in claims))
    add_check("actual_user_mismatch", sum(
        source_map[r["window_id"]]["actual_user_id"] != r["actual_user_id"] for r in claims
    ))
    add_check("genuine_flag_mismatch", sum(
        int(r["actual_user_id"] == r["claimed_user_id"]) != int(r["is_genuine"]) for r in claims
    ))
    add_check("wrong_split", sum(r["split"] != split for r in claims))

    by_claimed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in claims:
        by_claimed[r["claimed_user_id"]].append(r)
        by_window[r["window_id"]].append(r)

    for claimed in users:
        arr = by_claimed[claimed]
        genuine = sum(int(r["is_genuine"]) for r in arr)
        impostor = len(arr) - genuine
        counts.append({
            "cohort": cohort, "protocol": protocol, "split": split,
            "claimed_user_id": claimed,
            "genuine_claims": genuine,
            "impostor_claims": impostor,
            "total_claims": len(arr),
            "unique_actual_impostor_users": len({r["actual_user_id"] for r in arr if not int(r["is_genuine"])}),
            "unique_source_sessions": len({str(r["session_id"]) for r in arr}),
        })
        expected_genuine = sum(r["actual_user_id"] == claimed for r in source_rows)
        checks.append({
            "cohort": cohort, "protocol": protocol, "split": split,
            "check": f"{claimed}_genuine_count", "observed": genuine,
            "expected": expected_genuine,
            "status": "PASS" if genuine == expected_genuine else "FAIL",
        })
        if protocol == "balanced_1to1":
            checks.append({
                "cohort": cohort, "protocol": protocol, "split": split,
                "check": f"{claimed}_balanced_ratio", "observed": impostor,
                "expected": genuine,
                "status": "PASS" if impostor == genuine else "FAIL",
            })
        else:
            checks.append({
                "cohort": cohort, "protocol": protocol, "split": split,
                "check": f"{claimed}_all_windows", "observed": len(arr),
                "expected": len(source_rows),
                "status": "PASS" if len(arr) == len(source_rows) else "FAIL",
            })

    if protocol == "realistic_all_impostor":
        bad_window_claim_counts = sum(len(v) != len(users) for v in by_window.values())
        bad_window_genuine_counts = sum(sum(int(x["is_genuine"]) for x in v) != 1 for v in by_window.values())
        add_check("all_impostor_windows_wrong_claim_count", bad_window_claim_counts)
        add_check("all_impostor_windows_wrong_genuine_count", bad_window_genuine_counts)
        expected_total = len(source_rows) * len(users)
        checks.append({
            "cohort": cohort, "protocol": protocol, "split": split,
            "check": "all_impostor_total_claims", "observed": len(claims),
            "expected": expected_total,
            "status": "PASS" if len(claims) == expected_total else "FAIL",
        })
    else:
        expected_total = 2 * len(source_rows)
        checks.append({
            "cohort": cohort, "protocol": protocol, "split": split,
            "check": "balanced_total_claims", "observed": len(claims),
            "expected": expected_total,
            "status": "PASS" if len(claims) == expected_total else "FAIL",
        })
    return checks, counts


def write_manifest(root: Path) -> None:
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name == "output_manifest_sha256.csv":
            continue
        rows.append({
            "relative_path": p.relative_to(root).as_posix(),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    write_csv(root / "output_manifest_sha256.csv", rows,
              ["relative_path", "size_bytes", "sha256"])


def zip_output(output_dir: Path) -> Path:
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(output_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=(Path(output_dir.name) / p.relative_to(output_dir)).as_posix())
    return zip_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Keystroke–Mouse Stage 3A claim protocol pilot")
    ap.add_argument("--input-stage2c", required=True, help="Stage 2C review ZIP or extracted folder")
    ap.add_argument("--output", required=True, help="Output review folder")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing output folder/ZIP")
    args = ap.parse_args()

    input_path = Path(args.input_stage2c).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_zip = output_dir.with_suffix(".zip")
    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")
    if output_dir.exists() or output_zip.exists():
        if not args.overwrite:
            raise SystemExit("Output folder/ZIP already exists. Use --overwrite to replace it.")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if output_zip.exists():
            output_zip.unlink()
    output_dir.mkdir(parents=True)

    started = time.time()
    started_at = now_iso()
    log: list[str] = [
        f"[{started_at}] Stage 3A v{VERSION} started",
        f"Input: {input_path}",
        f"Output: {output_dir}",
    ]
    warnings: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    all_checks: list[dict[str, Any]] = []
    all_counts: list[dict[str, Any]] = []
    sampling_audit: list[dict[str, Any]] = []
    file_index: list[dict[str, Any]] = []
    source_counts: list[dict[str, Any]] = []

    try:
        with tempfile.TemporaryDirectory(prefix="km_stage3a_") as td:
            root = extract_input(input_path, Path(td))
            summary2c, users, user_to_idx = validate_stage2c(root)
            log.append(f"Validated Stage 2C root: {root}")
            log.append(f"Frozen users: {', '.join(users)}")

            for cohort in COHORTS:
                split_window_sets: dict[str, set[str]] = {}
                for split in SPLITS:
                    rows = load_index_and_validate(root, cohort, split, user_to_idx)
                    split_window_sets[split] = {r["window_id"] for r in rows}
                    source_counts.append({
                        "cohort": cohort, "split": split,
                        "source_windows": len(rows),
                        "source_sessions": len({str(r["session_id"]) for r in rows}),
                        "source_segments": len({str(r["segment_id"]) for r in rows}),
                        "users": len({r["actual_user_id"] for r in rows}),
                    })

                    balanced, audit = build_balanced_claims(rows, cohort, split, users, user_to_idx)
                    sampling_audit.extend(audit)
                    all_imp = build_all_impostor_claims(rows, cohort, split, users, user_to_idx)

                    for protocol, claims in (("balanced_1to1", balanced),
                                             ("realistic_all_impostor", all_imp)):
                        checks, counts = inspect_claims(claims, rows, cohort, protocol, split, users)
                        all_checks.extend(checks)
                        all_counts.extend(counts)
                        rel = Path("claims") / cohort / protocol / f"{split}_claims.csv"
                        out = output_dir / rel
                        write_csv(out, claims, CLAIM_FIELDS)
                        file_index.append({
                            "cohort": cohort, "protocol": protocol, "split": split,
                            "relative_path": rel.as_posix(),
                            "claims": len(claims),
                            "genuine_claims": sum(int(r["is_genuine"]) for r in claims),
                            "impostor_claims": sum(not int(r["is_genuine"]) for r in claims),
                            "sha256": sha256_file(out),
                        })
                        log.append(f"Created {rel.as_posix()}: {len(claims):,} claims")

                overlaps = {
                    "train_validation": len(split_window_sets["train"] & split_window_sets["validation"]),
                    "train_test": len(split_window_sets["train"] & split_window_sets["test"]),
                    "validation_test": len(split_window_sets["validation"] & split_window_sets["test"]),
                }
                for name, value in overlaps.items():
                    all_checks.append({
                        "cohort": cohort, "protocol": "all", "split": "cross_split",
                        "check": f"source_window_overlap_{name}", "observed": value,
                        "expected": 0, "status": "PASS" if value == 0 else "FAIL",
                    })

            # Low-count feasibility warnings are based only on frozen metadata, never scores/features.
            for row in all_counts:
                if row["protocol"] != "balanced_1to1":
                    continue
                g = int(row["genuine_claims"])
                if g < 5:
                    warnings.append({
                        "severity": "HIGH" if g < 3 else "MEDIUM",
                        "category": "low_genuine_claim_count",
                        "cohort": row["cohort"],
                        "split": row["split"],
                        "user_id": row["claimed_user_id"],
                        "message": (
                            f"Only {g} genuine windows are available. Personalized training/threshold "
                            "estimates will be unstable; retain this user but report a low-count sensitivity analysis."
                        ),
                    })

            failed_checks = [r for r in all_checks if r["status"] != "PASS"]
            if failed_checks:
                failures.extend({
                    "severity": "CRITICAL", "category": "integrity_check_failed",
                    "cohort": r["cohort"], "split": r["split"], "user_id": "",
                    "message": f"{r['protocol']} / {r['check']}: observed={r['observed']} expected={r['expected']}",
                } for r in failed_checks)

            write_csv(output_dir / "claim_file_index.csv", file_index,
                      ["cohort", "protocol", "split", "relative_path", "claims",
                       "genuine_claims", "impostor_claims", "sha256"])
            write_csv(output_dir / "source_window_counts.csv", source_counts,
                      ["cohort", "split", "source_windows", "source_sessions", "source_segments", "users"])
            write_csv(output_dir / "claim_counts_by_user_split_protocol.csv", all_counts,
                      ["cohort", "protocol", "split", "claimed_user_id", "genuine_claims",
                       "impostor_claims", "total_claims", "unique_actual_impostor_users",
                       "unique_source_sessions"])
            write_csv(output_dir / "impostor_sampling_audit.csv", sampling_audit,
                      ["cohort", "split", "claimed_user_id", "actual_impostor_user_id",
                       "available_windows", "selected_windows", "available_sessions",
                       "selected_sessions", "sampling_seed"])
            write_csv(output_dir / "claim_integrity_checks.csv", all_checks,
                      ["cohort", "protocol", "split", "check", "observed", "expected", "status"])
            write_csv(output_dir / "warnings_and_failures.csv", warnings + failures,
                      ["severity", "category", "cohort", "split", "user_id", "message"])

            schema_rows = [
                {"field": "claim_id", "role": "stable claim identifier", "model_input": "no"},
                {"field": "source_row_index", "role": "0-based row into the matching Stage 2C NPZ/index", "model_input": "no"},
                {"field": "window_id", "role": "frozen source window lineage", "model_input": "no"},
                {"field": "session_id", "role": "cluster/group identifier for leakage control and bootstrap", "model_input": "no"},
                {"field": "segment_id", "role": "idle-segment lineage", "model_input": "no"},
                {"field": "actual_user_id", "role": "ground-truth user", "model_input": "label only"},
                {"field": "claimed_user_id", "role": "identity being asserted", "model_input": "scoring target only"},
                {"field": "is_genuine", "role": "binary claim label", "model_input": "label only"},
                {"field": "sampling_role", "role": "genuine/sampled impostor/all impostor provenance", "model_input": "no"},
                {"field": "sampling_seed", "role": "reproducibility metadata", "model_input": "no"},
            ]
            write_csv(output_dir / "claim_schema.csv", schema_rows,
                      ["field", "role", "model_input"])

            frozen = {
                "stage": "Keystroke–Mouse Stage 3A",
                "version": VERSION,
                "base_seed": BASE_SEED,
                "input_scope": "frozen Stage 2C indexes and labels only",
                "cohorts": list(COHORTS),
                "splits": list(SPLITS),
                "users": users,
                "protocols": {
                    "balanced_1to1": {
                        "definition": "For each claimed user and split, include every genuine window and an equal number of unique impostor windows.",
                        "impostor_allocation": "capacity-aware near-equal allocation across the other users",
                        "within_user_sampling": "deterministic session-aware round robin without replacement",
                        "selection_inputs": ["cohort", "split", "claimed_user_id", "actual_user_id", "session_id", "window_id", "base_seed"],
                        "feature_values_used": False,
                    },
                    "realistic_all_impostor": {
                        "definition": "Every source window is evaluated once against every enrolled claimed user; each source window creates one genuine and ten impostor claims.",
                        "sampling": "none",
                    },
                },
                "frozen_stage2a_split_sha256": EXPECTED_STAGE2A_SPLIT_SHA256,
                "prohibitions": [
                    "No model training in Stage 3A",
                    "No calibration or threshold selection",
                    "No score-based sampling",
                    "No validation/test feature-driven selection",
                    "No moving windows between splits",
                    "No direct use of identifiers as predictive features",
                ],
            }
            (output_dir / "frozen_claim_protocol.json").write_text(
                json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            decisions = f"""# Keystroke–Mouse Stage 3A — Frozen Decisions

- Input is the validated Stage 2C artifact; its internal SHA-256 manifest is verified before use.
- Enrolled identities are User1 through User11, using the frozen Stage 2C mapping.
- Claims are model-agnostic and reference the corresponding Stage 2C row through `source_row_index`.
- Balanced protocol is exactly 1:1 per claimed user within each cohort and split.
- Every genuine window is retained in the balanced protocol.
- Balanced impostors are unique within a claimed-user/split/cohort table and are selected without replacement.
- Impostor allocation is capacity-aware and as even as possible across the other ten users.
- Selection within each actual impostor user is session-aware round robin with deterministic SHA-256 ranking.
- Realistic protocol evaluates every source window against all 11 claimed identities.
- Train, validation, and test claims are generated independently from their already-frozen source splits.
- Validation claims are reserved for calibration and threshold selection in a later stage.
- Test claims are reserved for final evaluation only.
- Overlapping windows from the same session remain statistically dependent; later confidence intervals and tests must cluster by session/user.
- Low-count user/split cells are not silently removed. They are flagged for sensitivity analysis.
- No training, scoring, calibration, thresholding, or metric computation is performed here.
"""
            (output_dir / "stage3a_frozen_decisions.md").write_text(decisions, encoding="utf-8")

            status = "FAIL" if failures else ("PASS_WITH_WARNINGS" if warnings else "PASS")
            finished_at = now_iso()
            elapsed = round(time.time() - started, 3)
            summary = {
                "stage": "Keystroke–Mouse Stage 3A",
                "version": VERSION,
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "elapsed_seconds": elapsed,
                "input_stage2c": {
                    "path": str(input_path),
                    "zip_sha256": sha256_file(input_path) if input_path.is_file() else None,
                    "stage2a_split_sha256": EXPECTED_STAGE2A_SPLIT_SHA256,
                    "stage2c_version": summary2c.get("version"),
                },
                "base_seed": BASE_SEED,
                "cohorts": list(COHORTS),
                "protocols": list(PROTOCOLS),
                "claim_files": len(file_index),
                "total_claims": sum(int(r["claims"]) for r in file_index),
                "total_genuine_claims": sum(int(r["genuine_claims"]) for r in file_index),
                "total_impostor_claims": sum(int(r["impostor_claims"]) for r in file_index),
                "warning_count": len(warnings),
                "failure_count": len(failures),
            }
            (output_dir / "stage3a_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            log += [
                f"Status: {status}",
                f"Claim files: {len(file_index)}",
                f"Total claims: {summary['total_claims']:,}",
                f"Warnings: {len(warnings)}",
                f"Failures: {len(failures)}",
                f"Elapsed seconds: {elapsed}",
                f"[{finished_at}] Stage 3A finished",
            ]
            (output_dir / "execution_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")

        write_manifest(output_dir)
        zip_path = zip_output(output_dir)
        print(f"Stage 3A status: {json.loads((output_dir/'stage3a_summary.json').read_text())['status']}")
        print(f"Output folder: {output_dir}")
        print(f"Output ZIP: {zip_path}")
        return 0 if not failures else 2
    except Exception as exc:
        failures.append({
            "severity": "CRITICAL", "category": "execution_exception",
            "cohort": "", "split": "", "user_id": "", "message": repr(exc),
        })
        write_csv(output_dir / "warnings_and_failures.csv", warnings + failures,
                  ["severity", "category", "cohort", "split", "user_id", "message"])
        log.append(f"FATAL: {repr(exc)}")
        (output_dir / "execution_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
        write_manifest(output_dir)
        zip_output(output_dir)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

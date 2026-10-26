#!/usr/bin/env python3
"""
Keystroke–Mouse Stage 3C Deep Models Pilot
CNN, BiLSTM, and CNN+BiLSTM per-user authentication models.

Methodological guarantees:
- Uses only frozen Stage 2C sequence tensors and masks.
- Trains one binary One-vs-Rest model per claimed user.
- Training uses Stage 2C Training only.
- Negatives are unique, deterministic, user-balanced and session-aware.
- Validation/Test claims are preserved row-for-row from Stage 3A.
- Produces raw logits only; higher means more genuine.
- No calibration, threshold selection, test metrics, or test-driven tuning.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install CPU PyTorch and dependencies wit<local_path>"
        "  py -3.12 -m pip install numpy scikit-learn psutil\n"
        "  py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        f"Original error: {exc}"
    )

try:
    import psutil
except ImportError:  # optional
    psutil = None

VERSION = "3.2.0"
STAGE = "Keystroke–Mouse Stage 3C Deep Models Pilot"
BASE_SEED = 31082026
COHORTS = ["keyboard", "mouse", "multimodal_primary", "multimodal_any"]
MODELS = ["cnn", "bilstm", "cnn_bilstm"]
PROTOCOLS = ["balanced_1to1", "realistic_all_impostor"]
SCORE_SPLITS = ["validation", "test"]
USERS = [f"User{i}" for i in range(1, 12)]
LOW_GENUINE_WARNING = 10
EPOCHS = 8
BATCH_SIZE = 128
SCORE_BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.15
GRAD_CLIP = 5.0
TORCH_THREADS = 2

CLAIM_REQUIRED = [
    "claim_id", "cohort", "protocol", "split", "source_row_index", "window_id",
    "session_id", "segment_id", "window_start_epoch", "window_end_epoch",
    "actual_user_id", "actual_user_index", "claimed_user_id", "claimed_user_index",
    "is_genuine", "sampling_role", "sampling_seed",
]
SCORE_APPEND = ["model_id", "model_family", "raw_score", "score_higher_is_genuine"]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_rank(*parts: Any) -> int:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def stable_seed32(*parts: Any) -> int:
    return stable_rank(BASE_SEED, *parts) % (2**31 - 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def remove_tree_with_retries(path: Path, *, fatal: bool) -> bool:
    gc.collect()
    last_error: Exception | None = None
    for delay in (0.0, 0.25, 0.75, 1.5, 3.0):
        if delay:
            time.sleep(delay)
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except (PermissionError, OSError) as exc:
            last_error = exc
            gc.collect()
    if fatal:
        raise RuntimeError(f"Could not remove directory after retries: {path}; {last_error}")
    return False


@contextmanager
def resilient_temp_workspace(prefix: str = "km_stage3c_"):
    path = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield path
    finally:
        remove_tree_with_retries(path, fatal=False)


def detect_root(base: Path, marker: str) -> Path:
    if (base / marker).exists():
        return base
    matches = list(base.rglob(marker))
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely locate {marker} under {base}; matches={len(matches)}")
    return matches[0].parent


def materialize_input(path: Path, marker: str, temp_base: Path, label: str) -> tuple[Path, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        return detect_root(path, marker), ""
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{label} must be a directory or ZIP: {path}")
    dest = temp_base / label
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt input ZIP member: {bad}")
        zf.extractall(dest)
    return detect_root(dest, marker), sha256_file(path)


def load_index_sessions(path: Path, expected_rows: int) -> list[str]:
    sessions: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "session_id" not in r.fieldnames:
            raise ValueError(f"Missing session_id in {path}")
        for row in r:
            sessions.append(row["session_id"])
    if len(sessions) != expected_rows:
        raise ValueError(f"Index/array mismatch: {path.name} rows={len(sessions)}, expected={expected_rows}")
    return sessions


def load_index_window_ids(path: Path, expected_rows: int) -> list[str]:
    ids: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "window_id" not in r.fieldnames:
            raise ValueError(f"Missing window_id in {path}")
        for row in r:
            ids.append(row["window_id"])
    if len(ids) != expected_rows:
        raise ValueError(f"Index/array mismatch: {path.name} rows={len(ids)}, expected={expected_rows}")
    return ids


def session_round_robin(indices: list[int], sessions: list[str], target: int, *rank_parts: Any) -> list[int]:
    if target >= len(indices):
        return sorted(indices)
    by_session: dict[str, list[int]] = defaultdict(list)
    for idx in indices:
        by_session[sessions[idx]].append(idx)
    session_order = sorted(by_session, key=lambda s: stable_rank(*rank_parts, "session", s))
    for s in session_order:
        by_session[s].sort(key=lambda i: stable_rank(*rank_parts, "window", s, i))
    pos = {s: 0 for s in session_order}
    selected: list[int] = []
    while len(selected) < target:
        progressed = False
        for s in session_order:
            p = pos[s]
            if p < len(by_session[s]):
                selected.append(by_session[s][p])
                pos[s] += 1
                progressed = True
                if len(selected) == target:
                    break
        if not progressed:
            raise RuntimeError("Session round-robin exhausted unexpectedly")
    return selected


def allocate_quotas(target: int, capacities: dict[int, int], *rank_parts: Any) -> dict[int, int]:
    if target > sum(capacities.values()):
        raise ValueError(f"Insufficient negative capacity: target={target}, capacity={sum(capacities.values())}")
    order = sorted(capacities, key=lambda u: stable_rank(*rank_parts, "user", u))
    quotas = {u: 0 for u in capacities}
    remain = target
    while remain:
        progressed = False
        for u in order:
            if quotas[u] < capacities[u]:
                quotas[u] += 1
                remain -= 1
                progressed = True
                if remain == 0:
                    break
        if not progressed:
            raise RuntimeError("Quota allocation stalled")
    return quotas


def select_balanced_negatives(y: np.ndarray, sessions: list[str], claimed: int,
                              target: int, cohort: str, model: str) -> list[int]:
    by_user: dict[int, list[int]] = {}
    for u in sorted(int(v) for v in np.unique(y) if int(v) != claimed):
        by_user[u] = np.flatnonzero(y == u).astype(int).tolist()
    capacities = {u: len(v) for u, v in by_user.items()}
    quotas = allocate_quotas(target, capacities, cohort, model, claimed, "negative_quota")
    selected: list[int] = []
    for u in sorted(by_user):
        selected.extend(session_round_robin(
            by_user[u], sessions, quotas[u], cohort, model, claimed, u, "negative_select"
        ))
    selected.sort(key=lambda i: stable_rank(cohort, model, claimed, "negative_final", i))
    return selected


class PeakRSSMonitor:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_mb = float("nan")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if psutil is None:
            return self
        proc = psutil.Process(os.getpid())
        self.peak_mb = proc.memory_info().rss / (1024**2)

        def poll() -> None:
            while not self._stop.wait(self.interval):
                try:
                    self.peak_mb = max(self.peak_mb, proc.memory_info().rss / (1024**2))
                except Exception:
                    return
        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


def masked_mean_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(dtype=x.dtype).unsqueeze(-1)
    count = m.sum(dim=1).clamp_min(1.0)
    mean = (x * m).sum(dim=1) / count
    masked = x.masked_fill(~mask.unsqueeze(-1), -1e9)
    mx = masked.max(dim=1).values
    empty = (mask.sum(dim=1) == 0).unsqueeze(-1)
    mx = torch.where(empty, torch.zeros_like(mx), mx)
    return torch.cat([mean, mx], dim=1)


class CNNBranch(nn.Module):
    out_dim = 64
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2)).transpose(1, 2)
        return masked_mean_max(h, mask)


class BiLSTMBranch(nn.Module):
    out_dim = 96
    def __init__(self, input_dim: int):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, 24, num_layers=1, batch_first=True, bidirectional=True)
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        return masked_mean_max(h, mask)


class CNNBiLSTMBranch(nn.Module):
    out_dim = 96
    def __init__(self, input_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.rnn = nn.LSTM(32, 24, num_layers=1, batch_first=True, bidirectional=True)
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.rnn(h)
        return masked_mean_max(h, mask)


def make_branch(family: str, input_dim: int) -> nn.Module:
    if family == "cnn":
        return CNNBranch(input_dim)
    if family == "bilstm":
        return BiLSTMBranch(input_dim)
    if family == "cnn_bilstm":
        return CNNBiLSTMBranch(input_dim)
    raise ValueError(family)


class DeepAuthModel(nn.Module):
    def __init__(self, family: str, cohort: str, keyboard_dim: int | None, mouse_dim: int | None):
        super().__init__()
        self.family = family
        self.cohort = cohort
        self.multimodal = cohort.startswith("multimodal")
        if self.multimodal:
            assert keyboard_dim is not None and mouse_dim is not None
            self.keyboard_branch = make_branch(family, keyboard_dim)
            self.mouse_branch = make_branch(family, mouse_dim)
            total = self.keyboard_branch.out_dim + self.mouse_branch.out_dim + 2
            self.head = nn.Sequential(nn.Linear(total, 48), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(48, 1))
        else:
            dim = keyboard_dim if cohort == "keyboard" else mouse_dim
            assert dim is not None
            self.branch = make_branch(family, dim)
            self.head = nn.Sequential(nn.Linear(self.branch.out_dim, 32), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(32, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.multimodal:
            ke = self.keyboard_branch(batch["keyboard_X"], batch["keyboard_mask"])
            me = self.mouse_branch(batch["mouse_X"], batch["mouse_mask"])
            mod = batch["modality_mask"].to(dtype=ke.dtype)
            ke = ke * mod[:, 0:1]
            me = me * mod[:, 1:2]
            z = torch.cat([ke, me, mod], dim=1)
        else:
            z = self.branch(batch["X"], batch["mask"])
        return self.head(z).squeeze(1)


@dataclass
class CohortData:
    cohort: str
    arrays: dict[str, np.ndarray]
    y: np.ndarray
    sessions: list[str]
    window_ids: list[str]

    def size(self) -> int:
        return int(len(self.y))

    def subset_torch(self, indices: np.ndarray) -> dict[str, torch.Tensor]:
        if self.cohort.startswith("multimodal"):
            return {
                "keyboard_X": torch.from_numpy(self.arrays["keyboard_X"][indices]).float(),
                "keyboard_mask": torch.from_numpy(self.arrays["keyboard_mask"][indices]).bool(),
                "mouse_X": torch.from_numpy(self.arrays["mouse_X"][indices]).float(),
                "mouse_mask": torch.from_numpy(self.arrays["mouse_mask"][indices]).bool(),
                "modality_mask": torch.from_numpy(self.arrays["modality_mask"][indices]).float(),
            }
        return {
            "X": torch.from_numpy(self.arrays["X"][indices]).float(),
            "mask": torch.from_numpy(self.arrays["mask"][indices]).bool(),
        }


def load_cohort_data(root2c: Path, cohort: str, split: str) -> CohortData:
    p = root2c / "sequences" / f"{cohort}_{split}.npz"
    arrays: dict[str, np.ndarray] = {}
    with np.load(p, allow_pickle=False) as z:
        y = np.array(z["y_user"], dtype=np.int16, copy=True)
        if cohort.startswith("multimodal"):
            for k in ["keyboard_X", "keyboard_mask", "mouse_X", "mouse_mask", "modality_mask"]:
                arrays[k] = np.array(z[k], copy=True)
        else:
            arrays["X"] = np.array(z["X"], copy=True)
            arrays["mask"] = np.array(z["mask"], copy=True)
    idx = root2c / "indexes" / f"{cohort}_sequence_{split}_index.csv"
    sessions = load_index_sessions(idx, len(y))
    window_ids = load_index_window_ids(idx, len(y))
    for k, v in arrays.items():
        if v.shape[0] != len(y):
            raise ValueError(f"Array row mismatch {cohort}/{split}/{k}")
        if k.endswith("_X") or k == "X":
            if not np.isfinite(v).all():
                raise ValueError(f"Non-finite sequence array {cohort}/{split}/{k}")
    return CohortData(cohort, arrays, y, sessions, window_ids)


def model_dims(data: CohortData) -> tuple[int | None, int | None]:
    if data.cohort == "keyboard":
        return int(data.arrays["X"].shape[2]), None
    if data.cohort == "mouse":
        return None, int(data.arrays["X"].shape[2])
    return int(data.arrays["keyboard_X"].shape[2]), int(data.arrays["mouse_X"].shape[2])


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def deterministic_epoch_order(n: int, seed: int, epoch: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed32(seed, "epoch", epoch))
    return rng.permutation(n).astype(np.int64)


def train_one_model(family: str, data: CohortData, claimed: int, artifact_dir: Path,
                    warnings_rows: list[dict[str, Any]], history_rows: list[dict[str, Any]]) -> tuple[DeepAuthModel, dict[str, Any], dict[str, Any]]:
    cohort = data.cohort
    user_id = USERS[claimed]
    y = data.y
    pos = np.flatnonzero(y == claimed).astype(int).tolist()
    if not pos:
        raise ValueError(f"No positive training windows for {cohort}/{user_id}")
    if len(pos) < LOW_GENUINE_WARNING:
        warnings_rows.append({
            "severity": "HIGH", "code": "LOW_TRAIN_GENUINE", "cohort": cohort,
            "model_family": family, "user_id": user_id,
            "message": f"Only {len(pos)} genuine training windows; deep model estimates may be unstable.",
        })
    pos_sessions = {data.sessions[i] for i in pos}
    if len(pos_sessions) < 2:
        warnings_rows.append({
            "severity": "HIGH", "code": "SINGLE_POSITIVE_SESSION", "cohort": cohort,
            "model_family": family, "user_id": user_id,
            "message": "Genuine training windows originate from one session only.",
        })
    neg = select_balanced_negatives(y, data.sessions, claimed, len(pos), cohort, family)
    idx = np.asarray(pos + neg, dtype=np.int64)
    target = np.concatenate([np.ones(len(pos), dtype=np.float32), np.zeros(len(neg), dtype=np.float32)])
    base_order = np.argsort([stable_rank(cohort, family, claimed, "base_train_order", int(i)) for i in idx])
    idx, target = idx[base_order], target[base_order]

    seed = stable_seed32(cohort, family, claimed)
    seed_everything(seed)
    kd, md = model_dims(data)
    model = DeepAuthModel(family, cohort, kd, md)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    start = time.perf_counter()

    for epoch in range(EPOCHS):
        order = deterministic_epoch_order(len(idx), seed, epoch)
        total_loss = 0.0
        total_n = 0
        for start_i in range(0, len(order), BATCH_SIZE):
            loc = order[start_i:start_i + BATCH_SIZE]
            batch_idx = idx[loc]
            batch = data.subset_torch(batch_idx)
            yy = torch.from_numpy(target[loc]).float()
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = loss_fn(logits, yy)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss: {cohort}/{family}/{user_id}/epoch{epoch+1}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += float(loss.detach()) * len(loc)
            total_n += len(loc)
        mean_loss = total_loss / max(1, total_n)
        history_rows.append({
            "cohort": cohort, "model_family": family, "user_id": user_id,
            "epoch": epoch + 1, "training_rows": len(idx), "mean_bce_loss": f"{mean_loss:.12g}",
            "validation_used_for_training": 0,
        })

    elapsed = time.perf_counter() - start
    model.eval()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / f"{user_id}.pt"
    torch.save({
        "stage": STAGE, "version": VERSION, "family": family, "cohort": cohort,
        "claimed_user_id": user_id, "claimed_user_index": claimed,
        "keyboard_dim": kd, "mouse_dim": md,
        "state_dict": model.state_dict(),
        "hyperparameters": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT, "grad_clip": GRAD_CLIP,
            "seed": seed,
        },
    }, artifact)

    audit = {
        "cohort": cohort, "model_family": family, "user_id": user_id,
        "claimed_user_index": claimed, "genuine_available": len(pos), "genuine_used": len(pos),
        "impostor_used": len(neg), "training_rows": len(idx),
        "positive_sessions_available": len(pos_sessions), "positive_sessions_used": len(pos_sessions),
        "negative_users_used": len({int(y[i]) for i in neg}),
        "negative_sessions_used": len({data.sessions[i] for i in neg}),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "random_seed": seed,
        "validation_used_for_training": 0,
    }
    resource = {
        "cohort": cohort, "model_family": family, "user_id": user_id,
        "train_seconds": round(elapsed, 6), "model_size_bytes": artifact.stat().st_size,
        "parameter_count": count_parameters(model),
        "parameter_count_definition": "trainable and non-trainable tensor elements",
    }
    return model, audit, resource


@torch.no_grad()
def score_model(model: DeepAuthModel, data: CohortData) -> np.ndarray:
    model.eval()
    out = np.empty(data.size(), dtype=np.float64)
    for start_i in range(0, data.size(), SCORE_BATCH_SIZE):
        end_i = min(data.size(), start_i + SCORE_BATCH_SIZE)
        idx = np.arange(start_i, end_i, dtype=np.int64)
        batch = data.subset_torch(idx)
        logits = model(batch)
        out[start_i:end_i] = logits.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(out).all():
        raise ValueError("Non-finite model scores")
    return out


def write_claim_scores(claim_path: Path, out_path: Path, score_matrix: np.ndarray,
                       family: str, cohort: str, protocol: str, split: str,
                       diagnostic_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duplicate_ids = 0
    seen: set[str] = set()
    finite_count = 0
    total = 0
    y_global: list[int] = []
    s_global: list[float] = []
    by_user_y: dict[str, list[int]] = defaultdict(list)
    by_user_s: dict[str, list[float]] = defaultdict(list)

    with claim_path.open(newline="", encoding="utf-8-sig") as fi, out_path.open("w", newline="", encoding="utf-8-sig") as fo:
        r = csv.DictReader(fi)
        if not r.fieldnames or any(c not in r.fieldnames for c in CLAIM_REQUIRED):
            raise ValueError(f"Claim schema mismatch: {claim_path}")
        fields = list(r.fieldnames) + SCORE_APPEND
        w = csv.DictWriter(fo, fieldnames=fields)
        w.writeheader()
        for row in r:
            total += 1
            cid = row["claim_id"]
            if cid in seen:
                duplicate_ids += 1
            seen.add(cid)
            src = int(row["source_row_index"])
            claimed = int(row["claimed_user_index"])
            if src < 0 or src >= score_matrix.shape[0] or claimed < 0 or claimed >= score_matrix.shape[1]:
                raise IndexError(f"Claim index out of bounds in {claim_path}: src={src}, claimed={claimed}")
            score = float(score_matrix[src, claimed])
            if math.isfinite(score):
                finite_count += 1
            score_text = format(score, ".12g")
            persisted = float(score_text)
            row.update({
                "model_id": f"{family}__{cohort}__{row['claimed_user_id']}",
                "model_family": family,
                "raw_score": score_text,
                "score_higher_is_genuine": 1,
            })
            w.writerow(row)
            if split == "validation":
                yy = int(row["is_genuine"])
                uid = row["claimed_user_id"]
                y_global.append(yy); s_global.append(persisted)
                by_user_y[uid].append(yy); by_user_s[uid].append(persisted)

    if split == "validation":
        user_aucs: list[float] = []
        for uid in USERS:
            yy, ss = by_user_y[uid], by_user_s[uid]
            auc = float("nan")
            if len(set(yy)) == 2:
                auc = float(roc_auc_score(yy, ss)); user_aucs.append(auc)
            diagnostic_rows.append({
                "cohort": cohort, "model_family": family, "protocol": protocol,
                "scope": "user", "claimed_user_id": uid,
                "genuine_claims": sum(yy), "impostor_claims": len(yy) - sum(yy),
                "validation_auc_raw": "" if not math.isfinite(auc) else f"{auc:.12g}",
                "test_metrics_computed": 0,
            })
        global_auc = float(roc_auc_score(y_global, s_global)) if len(set(y_global)) == 2 else float("nan")
        macro_auc = float(np.mean(user_aucs)) if user_aucs else float("nan")
        diagnostic_rows.extend([
            {"cohort": cohort, "model_family": family, "protocol": protocol,
             "scope": "global_claim_weighted", "claimed_user_id": "__GLOBAL__",
             "genuine_claims": sum(y_global), "impostor_claims": len(y_global)-sum(y_global),
             "validation_auc_raw": "" if not math.isfinite(global_auc) else f"{global_auc:.12g}",
             "test_metrics_computed": 0},
            {"cohort": cohort, "model_family": family, "protocol": protocol,
             "scope": "macro_user", "claimed_user_id": "__MACRO_USER__",
             "genuine_claims": sum(y_global), "impostor_claims": len(y_global)-sum(y_global),
             "validation_auc_raw": "" if not math.isfinite(macro_auc) else f"{macro_auc:.12g}",
             "test_metrics_computed": 0},
        ])

    return {
        "cohort": cohort, "model_family": family, "protocol": protocol, "split": split,
        "relative_path": str(out_path), "claims": total, "finite_scores": finite_count,
        "duplicate_claim_ids": duplicate_ids, "sha256": sha256_file(out_path),
    }


def make_manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "output_manifest_sha256.csv":
            rows.append({"relative_path": p.relative_to(root).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    return rows


def verify_manifest(root: Path, rows: list[dict[str, Any]]) -> None:
    bad = []
    for r in rows:
        p = root / str(r["relative_path"])
        if not p.exists() or p.stat().st_size != int(r["size_bytes"]) or sha256_file(p) != r["sha256"]:
            bad.append(r["relative_path"])
    if bad:
        raise RuntimeError(f"Manifest verification failed: {bad[:10]}")


def zip_directory(root: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zf:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(Path(root.name) / p.relative_to(root)))


def verify_zip(zip_path: Path, expected_files: int) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
        actual = sum(1 for n in zf.namelist() if not n.endswith("/"))
    if actual != expected_files:
        raise RuntimeError(f"ZIP count mismatch expected={expected_files} actual={actual}")


def main() -> int:
    ap = argparse.ArgumentParser(description=STAGE)
    ap.add_argument("--input-stage2c", required=True, help="Stage 2C review ZIP or directory")
    ap.add_argument("--input-stage3a", required=True, help="Stage 3A review ZIP or directory")
    ap.add_argument("--output", required=True, help="Output directory; ZIP is also created")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    seed_everything(BASE_SEED)
    started_at = now_iso(); t0 = time.perf_counter()
    input2c = Path(args.input_stage2c).expanduser().resolve()
    input3a = Path(args.input_stage3a).expanduser().resolve()
    out = Path(args.output).expanduser().resolve(); out_zip = out.with_suffix(".zip")
    if out.exists() or out_zip.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists. Use --overwrite: {out} or {out_zip}")
        if out.exists(): remove_tree_with_retries(out, fatal=True)
        if out_zip.exists(): out_zip.unlink()
    out.mkdir(parents=True)

    warnings_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    score_index_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    execution_lines = [f"{STAGE} v{VERSION}", f"Started: {started_at}"]

    with resilient_temp_workspace() as temp:
        root2c, zip2c_sha = materialize_input(input2c, "stage2c_summary.json", temp, "stage2c")
        root3a, zip3a_sha = materialize_input(input3a, "stage3a_summary.json", temp, "stage3a")
        s2 = load_json(root2c / "stage2c_summary.json"); s3 = load_json(root3a / "stage3a_summary.json")
        split2 = s2.get("input_stage2b", {}).get("stage2a_split_sha256", "")
        split3 = s3.get("input_stage2c", {}).get("stage2a_split_sha256", "")
        if not split2 or split2 != split3:
            raise ValueError(f"Frozen split mismatch Stage2C={split2} Stage3A={split3}")
        if s2.get("failure_count", 0) or s3.get("failure_count", 0):
            raise ValueError("Input stage reports failures")

        protocol = {
            "stage": STAGE, "version": VERSION, "base_seed": BASE_SEED,
            "cohorts": COHORTS, "model_families": MODELS,
            "training": {
                "scheme": "per-user binary One-vs-Rest",
                "samples": "all genuine plus equal number of unique deterministic user/session-balanced impostors",
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                "dropout": DROPOUT, "gradient_clip": GRAD_CLIP,
                "training_data": "Stage 2C Training only", "validation_used_for_training": False,
            },
            "architectures": {
                "cnn": "two Conv1D layers per modality branch + masked mean/max pooling",
                "bilstm": "one bidirectional LSTM layer per modality branch + masked mean/max pooling",
                "cnn_bilstm": "two Conv1D layers followed by bidirectional LSTM per modality branch",
                "multimodal_fusion": "late embedding concatenation plus modality-observation mask",
            },
            "score_orientation": "raw binary logit; higher means more genuine",
            "prohibitions": ["No calibration", "No threshold", "No test metrics", "No test-driven tuning", "No claim resampling"],
            "inputs": {"stage2c_zip_sha256": zip2c_sha, "stage3a_zip_sha256": zip3a_sha, "frozen_stage2a_split_sha256": split2},
        }
        write_json(out / "frozen_deep_model_protocol.json", protocol)
        execution_lines += [f"Stage2C root: {root2c}", f"Stage3A root: {root3a}", f"Frozen split: {split2}", f"Torch: {torch.__version__}"]

        all_score_seconds: dict[tuple[str, str], float] = defaultdict(float)
        all_score_claims: dict[tuple[str, str], int] = defaultdict(int)

        with PeakRSSMonitor() as peak:
            for cohort in COHORTS:
                train = load_cohort_data(root2c, cohort, "train")
                val = load_cohort_data(root2c, cohort, "validation")
                test = load_cohort_data(root2c, cohort, "test")
                if sorted(np.unique(train.y).tolist()) != list(range(11)):
                    raise ValueError(f"Unexpected labels in {cohort}")

                for family in MODELS:
                    models: dict[int, DeepAuthModel] = {}
                    for claimed in range(11):
                        model, audit, resource = train_one_model(
                            family, train, claimed, out / "models" / cohort / family,
                            warnings_rows, history_rows,
                        )
                        models[claimed] = model
                        training_rows.append(audit); resource_rows.append(resource)

                    score_matrices: dict[str, np.ndarray] = {}
                    for split, data in [("validation", val), ("test", test)]:
                        matrix = np.empty((data.size(), 11), dtype=np.float64)
                        st = time.perf_counter()
                        for claimed in range(11):
                            matrix[:, claimed] = score_model(models[claimed], data)
                        sec = time.perf_counter() - st
                        score_matrices[split] = matrix
                        resource_rows.append({
                            "cohort": cohort, "model_family": family, "user_id": "__ALL_USERS__",
                            "train_seconds": "", "model_size_bytes": "", "parameter_count": "",
                            "parameter_count_definition": "", "score_split": split,
                            "score_windows": data.size(), "score_user_models": 11,
                            "score_seconds": round(sec, 6),
                            "score_ms_per_window_user_pair": round(sec * 1000 / (data.size()*11), 9),
                        })

                    for protocol_name in PROTOCOLS:
                        for split in SCORE_SPLITS:
                            claim_path = root3a / "claims" / cohort / protocol_name / f"{split}_claims.csv"
                            out_path = out / "scores" / cohort / family / protocol_name / f"{split}_claim_scores.csv"
                            st = time.perf_counter()
                            idx_row = write_claim_scores(claim_path, out_path, score_matrices[split], family, cohort, protocol_name, split, diagnostic_rows)
                            sec = time.perf_counter() - st
                            idx_row["relative_path"] = out_path.relative_to(out).as_posix()
                            score_index_rows.append(idx_row)
                            all_score_seconds[(cohort, family)] += sec
                            all_score_claims[(cohort, family)] += int(idx_row["claims"])
                            integrity_rows.extend([
                                {"cohort": cohort, "model_family": family, "protocol": protocol_name, "split": split,
                                 "check": "row_count_positive", "value": idx_row["claims"], "expected": ">0",
                                 "status": "PASS" if idx_row["claims"] > 0 else "FAIL"},
                                {"cohort": cohort, "model_family": family, "protocol": protocol_name, "split": split,
                                 "check": "all_scores_finite", "value": idx_row["finite_scores"], "expected": idx_row["claims"],
                                 "status": "PASS" if idx_row["finite_scores"] == idx_row["claims"] else "FAIL"},
                                {"cohort": cohort, "model_family": family, "protocol": protocol_name, "split": split,
                                 "check": "duplicate_claim_ids", "value": idx_row["duplicate_claim_ids"], "expected": 0,
                                 "status": "PASS" if idx_row["duplicate_claim_ids"] == 0 else "FAIL"},
                            ])
                    models.clear(); score_matrices.clear(); gc.collect()
                del train, val, test; gc.collect()
            peak_rss_mb = peak.peak_mb

        for (cohort, family), seconds in sorted(all_score_seconds.items()):
            claims = all_score_claims[(cohort, family)]
            resource_rows.append({
                "cohort": cohort, "model_family": family, "user_id": "__CLAIM_SCORE_OUTPUT__",
                "train_seconds": "", "model_size_bytes": "", "parameter_count": "",
                "parameter_count_definition": "", "score_split": "validation+test;both_protocols",
                "score_windows": "", "score_user_models": 11, "score_seconds": round(seconds, 6),
                "score_ms_per_claim_output": round(seconds*1000/claims, 9) if claims else "", "claims_written": claims,
            })

        failed = [r for r in integrity_rows if r["status"] != "PASS"]
        for r in failed:
            failure_rows.append({"severity": "CRITICAL", "code": "INTEGRITY_CHECK_FAILED", "cohort": r["cohort"],
                                 "model_family": r["model_family"], "user_id": "", "message": json.dumps(r, ensure_ascii=False)})

        write_csv(out / "training_sample_audit.csv", training_rows, [
            "cohort", "model_family", "user_id", "claimed_user_index", "genuine_available", "genuine_used",
            "impostor_used", "training_rows", "positive_sessions_available", "positive_sessions_used",
            "negative_users_used", "negative_sessions_used", "epochs", "batch_size", "random_seed", "validation_used_for_training",
        ])
        write_csv(out / "training_history.csv", history_rows, [
            "cohort", "model_family", "user_id", "epoch", "training_rows", "mean_bce_loss", "validation_used_for_training",
        ])
        write_csv(out / "model_resource_metrics.csv", resource_rows, [
            "cohort", "model_family", "user_id", "train_seconds", "model_size_bytes", "parameter_count",
            "parameter_count_definition", "score_split", "score_windows", "score_user_models", "score_seconds",
            "score_ms_per_window_user_pair", "score_ms_per_claim_output", "claims_written",
        ])
        write_csv(out / "score_file_index.csv", score_index_rows, [
            "cohort", "model_family", "protocol", "split", "relative_path", "claims", "finite_scores", "duplicate_claim_ids", "sha256",
        ])
        write_csv(out / "score_integrity_checks.csv", integrity_rows, [
            "cohort", "model_family", "protocol", "split", "check", "value", "expected", "status",
        ])
        write_csv(out / "validation_auc_diagnostics.csv", diagnostic_rows, [
            "cohort", "model_family", "protocol", "scope", "claimed_user_id", "genuine_claims", "impostor_claims",
            "validation_auc_raw", "test_metrics_computed",
        ])
        write_csv(out / "warnings_and_failures.csv", warnings_rows + failure_rows, [
            "severity", "code", "cohort", "model_family", "user_id", "message",
        ])

        decisions = f"""# Stage 3C Deep Models Pilot — Frozen Decisions\n\n- Version: {VERSION}\n- Frozen split SHA-256: `{split2}`\n- Base seed: `{BASE_SEED}`\n- Architectures: CNN, BiLSTM, CNN+BiLSTM.\n- Models: one binary model per user, cohort, and architecture.\n- Training: Stage 2C Training only; all genuine plus equal unique impostor windows.\n- Epochs: {EPOCHS}; batch size: {BATCH_SIZE}; optimizer: AdamW; learning rate: {LEARNING_RATE}.\n- Validation is not used to fit weights or stop epochs.\n- Multimodal fusion: separate keyboard/mouse sequence branches with late embedding fusion and modality masks.\n- Raw output: binary logit; higher means more genuine.\n- Validation AUC is diagnostic only. No calibration, thresholds, test metrics, or test-driven selection.\n"""
        (out / "stage3c_frozen_decisions.md").write_text(decisions, encoding="utf-8")

        finished_at = now_iso(); elapsed = time.perf_counter() - t0
        status = "FAIL" if failure_rows else ("PASS_WITH_WARNINGS" if warnings_rows else "PASS")
        summary = {
            "stage": STAGE, "version": VERSION, "status": status,
            "started_at": started_at, "finished_at": finished_at, "elapsed_seconds": round(elapsed, 3),
            "inputs": {"stage2c": str(input2c), "stage2c_zip_sha256": zip2c_sha,
                       "stage3a": str(input3a), "stage3a_zip_sha256": zip3a_sha,
                       "frozen_stage2a_split_sha256": split2},
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                            "device": "cpu", "torch_threads": TORCH_THREADS, "psutil": getattr(psutil, "__version__", "not_installed")},
            "cohorts": COHORTS, "model_families": MODELS,
            "trained_user_models": len(COHORTS)*len(MODELS)*len(USERS),
            "score_files": len(score_index_rows), "total_claim_scores": sum(int(r["claims"]) for r in score_index_rows),
            "epochs": EPOCHS, "validation_used_for_training": False,
            "validation_diagnostics_only": True, "test_metrics_computed": False,
            "warning_count": len(warnings_rows), "failure_count": len(failure_rows),
            "failed_integrity_checks": len(failed),
            "peak_process_rss_mb": None if not math.isfinite(peak_rss_mb) else round(peak_rss_mb, 3),
        }
        write_json(out / "stage3c_summary.json", summary)
        execution_lines += [f"Finished: {finished_at}", f"Elapsed seconds: {elapsed:.3f}", f"Status: {status}",
                            f"Warnings: {len(warnings_rows)}", f"Failures: {len(failure_rows)}",
                            f"Models: {summary['trained_user_models']}", f"Score files: {len(score_index_rows)}",
                            f"Claim scores: {summary['total_claim_scores']}", f"Peak RSS MB: {summary['peak_process_rss_mb']}"]
        (out / "execution_log.txt").write_text("\n".join(execution_lines) + "\n", encoding="utf-8")

    (out / "STAGE3C_COMPLETE.ok").write_text(f"{STAGE} v{VERSION} completed at {now_iso()}\n", encoding="utf-8")
    manifest = make_manifest(out)
    write_csv(out / "output_manifest_sha256.csv", manifest, ["relative_path", "size_bytes", "sha256"])
    verify_manifest(out, manifest)
    zip_directory(out, out_zip)
    verify_zip(out_zip, len(manifest)+1)
    print(json.dumps({
        "status": load_json(out / "stage3c_summary.json")["status"],
        "output_directory": str(out), "output_zip": str(out_zip), "zip_size_bytes": out_zip.stat().st_size,
        "manifest_files_verified": len(manifest), "zip_crc_verified": True,
    }, ensure_ascii=False, indent=2))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Keystroke-Mouse Stage 4A v4.1.0
Validation-Only Temperature Calibration, Threshold Policy and Model Eligibility Audit.

Inputs:
  - Stage 3B classical review ZIP
  - Stage 3C deep review ZIP
Output:
  - Calibrator and threshold policy frozen from BALANCED VALIDATION only
  - Calibrated Validation and Test claim-level files
  - Validation-only diagnostics
  - NO aggregate Test metrics

Safety:
  - Reads ZIPs directly (no temporary extraction / no Windows NPZ locks)
  - Never changes input ZIPs
  - Never fits calibration/thresholds on Test
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

VERSION = "4.1.0"
FIT_PROTOCOL = "balanced_1to1"
PERSONAL_MIN_GENUINE = 20
PERSONAL_MIN_IMPOSTOR = 20
PERSONAL_MIN_IMPOSTOR_IDENTITIES = 5
EPS = 1e-12


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_float(x: str) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"Non-finite score: {x}")
    return v


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    a = np.asarray(z, dtype=np.float64)
    a = np.clip(a, -50.0, 50.0)
    out = 1.0 / (1.0 + np.exp(-a))
    return out if isinstance(z, np.ndarray) else float(out)


def make_user_equal_class_weights(users: Sequence[str], y: np.ndarray) -> np.ndarray:
    users_a = np.asarray(users, dtype=object)
    y = np.asarray(y, dtype=np.int8)
    w = np.zeros(len(y), dtype=np.float64)
    for u in sorted(set(users_a.tolist())):
        m = users_a == u
        pos = m & (y == 1)
        neg = m & (y == 0)
        np_, nn_ = int(pos.sum()), int(neg.sum())
        if np_ > 0:
            w[pos] = 0.5 / np_
        if nn_ > 0:
            w[neg] = 0.5 / nn_
    sw = float(w.sum())
    return w / sw if sw > 0 else np.full(len(y), 1.0 / max(len(y), 1))


def make_class_balanced_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int8)
    w = np.zeros(len(y), dtype=np.float64)
    p = y == 1
    n = y == 0
    if p.sum():
        w[p] = 0.5 / p.sum()
    if n.sum():
        w[n] = 0.5 / n.sum()
    sw = float(w.sum())
    return w / sw if sw > 0 else np.full(len(y), 1.0 / max(len(y), 1))


def weighted_nll(raw: np.ndarray, y: np.ndarray, w: np.ndarray, log_t: float) -> float:
    t = math.exp(log_t)
    p = np.asarray(sigmoid(raw / t), dtype=np.float64)
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(np.sum(w * (-(y * np.log(p) + (1 - y) * np.log(1 - p)))))


def fit_temperature(raw: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, float, float, bool]:
    raw = np.asarray(raw, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if len(raw) == 0 or len(np.unique(y)) < 2 or float(np.std(raw)) < 1e-12:
        before = weighted_nll(raw, y, w, 0.0) if len(raw) else float("nan")
        return 1.0, before, before, False
    lo, hi = -8.0, 8.0
    best_log_t = 0.0
    best_loss = float("inf")
    at_boundary = False
    for _ in range(5):
        grid = np.linspace(lo, hi, 161)
        losses = np.array([weighted_nll(raw, y, w, float(g)) for g in grid])
        i = int(np.argmin(losses))
        best_log_t = float(grid[i])
        best_loss = float(losses[i])
        at_boundary = i in (0, len(grid) - 1)
        step = (hi - lo) / (len(grid) - 1)
        lo, hi = best_log_t - 2.0 * step, best_log_t + 2.0 * step
        lo, hi = max(lo, -8.0), min(hi, 8.0)
    before = weighted_nll(raw, y, w, 0.0)
    return math.exp(best_log_t), before, best_loss, at_boundary


def weighted_auc(y: np.ndarray, score: np.ndarray, w: np.ndarray | None = None) -> float:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return float("nan")
    if w is None:
        w = np.ones(len(y), dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64)
    # Weighted Mann-Whitney with tie handling.
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    yy = y[order]
    ww = w[order]
    total_pos = float(ww[yy == 1].sum())
    total_neg = float(ww[yy == 0].sum())
    if total_pos <= 0 or total_neg <= 0:
        return float("nan")
    cum_neg = 0.0
    concord = 0.0
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        gp = float(ww[i:j][yy[i:j] == 1].sum())
        gn = float(ww[i:j][yy[i:j] == 0].sum())
        concord += gp * (cum_neg + 0.5 * gn)
        cum_neg += gn
        i = j
    return concord / (total_pos * total_neg)


def calibration_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray, bins: int = 10) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    w = np.asarray(w, dtype=np.float64)
    w = w / max(float(w.sum()), EPS)
    nll = float(np.sum(w * (-(y * np.log(p) + (1 - y) * np.log(1 - p)))))
    brier = float(np.sum(w * (p - y) ** 2))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        if i == bins - 1:
            m = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            m = (p >= edges[i]) & (p < edges[i + 1])
        if not np.any(m):
            continue
        wm = w[m]
        mass = float(wm.sum())
        conf = float(np.sum(wm * p[m]) / mass)
        acc = float(np.sum(wm * y[m]) / mass)
        ece += mass * abs(acc - conf)
    return {"nll": nll, "brier": brier, "ece10": float(ece)}


def select_eer_threshold(scores: np.ndarray, y: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    w = np.asarray(w, dtype=np.float64)
    pos_total = float(w[y == 1].sum())
    neg_total = float(w[y == 0].sum())
    if pos_total <= 0 or neg_total <= 0:
        return {"threshold": 0.5, "far": float("nan"), "frr": float("nan"), "eer": float("nan"), "balanced_error": float("nan")}
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    yy = y[order]
    ww = w[order]
    tp = fp = 0.0
    candidates = []
    max_s = float(np.max(s))
    candidates.append((math.nextafter(max_s, math.inf), 0.0, 1.0))
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        tp += float(ww[i:j][yy[i:j] == 1].sum())
        fp += float(ww[i:j][yy[i:j] == 0].sum())
        far = fp / neg_total
        frr = (pos_total - tp) / pos_total
        candidates.append((float(s[i]), far, frr))
        i = j
    best = min(candidates, key=lambda x: (abs(x[1] - x[2]), 0.5 * (x[1] + x[2]), -x[0]))
    t, far, frr = best
    return {"threshold": t, "far": far, "frr": frr, "eer": 0.5 * (far + frr), "balanced_error": 0.5 * (far + frr)}


def binary_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    tp = float(w[(y == 1) & (pred == 1)].sum())
    fn = float(w[(y == 1) & (pred == 0)].sum())
    fp = float(w[(y == 0) & (pred == 1)].sum())
    tn = float(w[(y == 0) & (pred == 0)].sum())
    far = fp / max(fp + tn, EPS)
    frr = fn / max(tp + fn, EPS)
    acc = (tp + tn) / max(tp + tn + fp + fn, EPS)
    precision = tp / max(tp + fp, EPS)
    recall = tp / max(tp + fn, EPS)
    f1 = 2 * precision * recall / max(precision + recall, EPS)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    auc = weighted_auc(y, score, w)
    return {"far": far, "frr": frr, "accuracy": acc, "f1": f1, "mcc": mcc, "auc": auc}


@dataclass
class ZipScoreSource:
    label: str
    zip_path: Path
    zf: zipfile.ZipFile
    root: str
    entries: List[Dict[str, str]]

    @classmethod
    def open(cls, label: str, path: Path) -> "ZipScoreSource":
        zf = zipfile.ZipFile(path, "r")
        matches = [n for n in zf.namelist() if n.endswith("score_file_index.csv")]
        if len(matches) != 1:
            zf.close()
            raise RuntimeError(f"{path}: expected one score_file_index.csv, found {len(matches)}")
        index_name = matches[0]
        root = index_name[: -len("score_file_index.csv")]
        with zf.open(index_name, "r") as fb:
            text = io.TextIOWrapper(fb, encoding="utf-8-sig", newline="")
            entries = list(csv.DictReader(text))
        return cls(label, path, zf, root, entries)

    def read_rows(self, relative_path: str) -> List[Dict[str, str]]:
        name = self.root + relative_path.replace("\\", "/")
        with self.zf.open(name, "r") as fb:
            text = io.TextIOWrapper(fb, encoding="utf-8-sig", newline="")
            return list(csv.DictReader(text))

    def iter_rows(self, relative_path: str):
        name = self.root + relative_path.replace("\\", "/")
        fb = self.zf.open(name, "r")
        text = io.TextIOWrapper(fb, encoding="utf-8-sig", newline="")
        reader = csv.DictReader(text)
        try:
            for row in reader:
                yield row, reader.fieldnames
        finally:
            text.close()

    def close(self):
        self.zf.close()


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def create_zip(folder: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(folder.parent).as_posix())
    with zipfile.ZipFile(zip_path, "r") as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC failure: {bad}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-stage3b", required=True)
    ap.add_argument("--input-stage3c", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    s3b = Path(args.input_stage3b).resolve()
    s3c = Path(args.input_stage3c).resolve()
    out = Path(args.output).resolve()
    out_zip = out.with_suffix(".zip")
    if not s3b.is_file() or not s3c.is_file():
        raise FileNotFoundError("Both Stage 3B and Stage 3C ZIP inputs are required.")
    if out.exists() or out_zip.exists():
        if not args.overwrite:
            raise FileExistsError("Output exists. Use --overwrite.")
        if out.exists():
            shutil.rmtree(out)
        if out_zip.exists():
            out_zip.unlink()
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    warnings: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    integrity: List[Dict[str, object]] = []
    calibrator_rows: List[Dict[str, object]] = []
    threshold_rows: List[Dict[str, object]] = []
    eligibility_rows: List[Dict[str, object]] = []
    calibration_diag: List[Dict[str, object]] = []
    threshold_diag: List[Dict[str, object]] = []
    score_index_out: List[Dict[str, object]] = []

    sources = [ZipScoreSource.open("stage3b", s3b), ZipScoreSource.open("stage3c", s3c)]
    try:
        all_entries = []
        for src in sources:
            for e in src.entries:
                x = dict(e)
                x["source_label"] = src.label
                x["source_obj"] = src
                all_entries.append(x)
        keys = [(e["cohort"], e["model_family"], e["protocol"], e["split"]) for e in all_entries]
        if len(keys) != len(set(keys)):
            raise RuntimeError("Duplicate cohort/model/protocol/split score inputs across Stage 3B and 3C.")
        models = sorted(set((e["cohort"], e["model_family"]) for e in all_entries))

        # Freeze calibrators and thresholds exclusively from BALANCED VALIDATION.
        policy: Dict[Tuple[str, str], Dict[str, object]] = {}
        for cohort, model in models:
            match = [e for e in all_entries if e["cohort"] == cohort and e["model_family"] == model and e["protocol"] == FIT_PROTOCOL and e["split"] == "validation"]
            if len(match) != 1:
                raise RuntimeError(f"Missing/duplicate balanced validation score file for {cohort}/{model}")
            e = match[0]
            rows = e["source_obj"].read_rows(e["relative_path"])
            if not rows:
                raise RuntimeError(f"Empty validation scores for {cohort}/{model}")
            direction_vals = {r.get("score_higher_is_genuine", "") for r in rows}
            if direction_vals != {"1"}:
                failures.append({"severity":"FAIL","code":"SCORE_DIRECTION_MISMATCH","cohort":cohort,"model_family":model,"claimed_user_id":"","message":f"Expected score_higher_is_genuine=1, got {sorted(direction_vals)}"})
                raise RuntimeError(f"Score direction mismatch for {cohort}/{model}")
            raw = np.array([safe_float(r["raw_score"]) for r in rows], dtype=np.float64)
            y = np.array([int(r["is_genuine"]) for r in rows], dtype=np.int8)
            users = [r["claimed_user_id"] for r in rows]
            actual_users = [r["actual_user_id"] for r in rows]
            global_w = make_user_equal_class_weights(users, y)
            gT, gnll0, gnll1, gbound = fit_temperature(raw, y, global_w)
            gp_before = np.asarray(sigmoid(raw), dtype=np.float64)
            gp_after = np.asarray(sigmoid(raw / gT), dtype=np.float64)
            gcal0 = calibration_metrics(y, gp_before, global_w)
            gcal1 = calibration_metrics(y, gp_after, global_w)
            gthr = select_eer_threshold(gp_after, y, global_w)
            global_id = f"TEMP_GLOBAL__{cohort}__{model}"
            calibrator_rows.append({
                "cohort":cohort,"model_family":model,"scope":"global","claimed_user_id":"ALL",
                "calibrator_id":global_id,"fit_protocol":FIT_PROTOCOL,"fit_split":"validation",
                "temperature":f"{gT:.12g}","genuine_claims":int((y==1).sum()),"impostor_claims":int((y==0).sum()),
                "impostor_identities":len(set(a for a,yy,u in zip(actual_users,y,users) if yy==0)),
                "eligible":1,"nll_before":f"{gnll0:.12g}","nll_after":f"{gnll1:.12g}","temperature_at_search_boundary":int(gbound)
            })
            threshold_rows.append({
                "cohort":cohort,"model_family":model,"scope":"global","claimed_user_id":"ALL",
                "threshold_id":f"EER_GLOBAL__{cohort}__{model}","threshold_source":"global_validation_eer",
                "fit_protocol":FIT_PROTOCOL,"fit_split":"validation","threshold":f"{gthr['threshold']:.12g}",
                "validation_far":f"{gthr['far']:.12g}","validation_frr":f"{gthr['frr']:.12g}","validation_eer":f"{gthr['eer']:.12g}",
                "genuine_claims":int((y==1).sum()),"impostor_claims":int((y==0).sum()),"eligible":1
            })
            calibration_diag.append({
                "cohort":cohort,"model_family":model,"scope":"global","claimed_user_id":"ALL",
                "temperature":f"{gT:.12g}","nll_before":f"{gcal0['nll']:.12g}","nll_after":f"{gcal1['nll']:.12g}",
                "brier_before":f"{gcal0['brier']:.12g}","brier_after":f"{gcal1['brier']:.12g}",
                "ece10_before":f"{gcal0['ece10']:.12g}","ece10_after":f"{gcal1['ece10']:.12g}"
            })
            personal = {}
            user_aucs = []
            user_auc_lt_half = 0
            for u in sorted(set(users), key=lambda x: int(x.replace("User",""))):
                m = np.array([x == u for x in users], dtype=bool)
                uraw, uy = raw[m], y[m]
                uactual = np.array(actual_users, dtype=object)[m]
                ug, ui = int((uy==1).sum()), int((uy==0).sum())
                imp_ids = len(set(uactual[uy==0].tolist()))
                eligible = ug >= PERSONAL_MIN_GENUINE and ui >= PERSONAL_MIN_IMPOSTOR and imp_ids >= PERSONAL_MIN_IMPOSTOR_IDENTITIES and float(np.std(uraw)) >= 1e-12
                uw = make_class_balanced_weights(uy)
                auc_raw = weighted_auc(uy, uraw, uw)
                if math.isfinite(auc_raw):
                    user_aucs.append(auc_raw)
                    if auc_raw < 0.5:
                        user_auc_lt_half += 1
                if eligible:
                    uT, unll0, unll1, ubound = fit_temperature(uraw, uy, uw)
                    cal_source = "personal_temperature"
                    cal_id = f"TEMP_PERSONAL__{cohort}__{model}__{u}"
                else:
                    uT, unll0, unll1, ubound = gT, float("nan"), float("nan"), False
                    cal_source = "global_temperature_fallback"
                    cal_id = global_id
                    warnings.append({"severity":"HIGH" if ug < 5 else "WARN","code":"PERSONAL_CALIBRATION_FALLBACK","cohort":cohort,"model_family":model,"claimed_user_id":u,"message":f"Personal calibration ineligible: genuine={ug}, impostor={ui}, impostor_identities={imp_ids}; global temperature used."})
                up = np.asarray(sigmoid(uraw / uT), dtype=np.float64)
                threshold_eligible = eligible
                if threshold_eligible:
                    uthr = select_eer_threshold(up, uy, uw)
                    th_source = "personal_validation_eer"
                    th_id = f"EER_PERSONAL__{cohort}__{model}__{u}"
                else:
                    uthr = dict(gthr)
                    th_source = "global_validation_eer_fallback"
                    th_id = f"EER_GLOBAL__{cohort}__{model}"
                    warnings.append({"severity":"HIGH" if ug < 5 else "WARN","code":"PERSONAL_THRESHOLD_FALLBACK","cohort":cohort,"model_family":model,"claimed_user_id":u,"message":f"Personal threshold ineligible: genuine={ug}, impostor={ui}, impostor_identities={imp_ids}; global EER threshold used."})
                personal[u] = {"temperature":uT,"calibrator_id":cal_id,"calibration_source":cal_source,"threshold":uthr["threshold"],"threshold_id":th_id,"threshold_source":th_source,"eligible":eligible}
                calibrator_rows.append({
                    "cohort":cohort,"model_family":model,"scope":"user","claimed_user_id":u,
                    "calibrator_id":cal_id,"fit_protocol":FIT_PROTOCOL,"fit_split":"validation","temperature":f"{uT:.12g}",
                    "genuine_claims":ug,"impostor_claims":ui,"impostor_identities":imp_ids,"eligible":int(eligible),
                    "nll_before":f"{unll0:.12g}" if math.isfinite(unll0) else "","nll_after":f"{unll1:.12g}" if math.isfinite(unll1) else "",
                    "temperature_at_search_boundary":int(ubound)
                })
                threshold_rows.append({
                    "cohort":cohort,"model_family":model,"scope":"user","claimed_user_id":u,
                    "threshold_id":th_id,"threshold_source":th_source,"fit_protocol":FIT_PROTOCOL,"fit_split":"validation",
                    "threshold":f"{uthr['threshold']:.12g}","validation_far":f"{uthr['far']:.12g}" if threshold_eligible else "",
                    "validation_frr":f"{uthr['frr']:.12g}" if threshold_eligible else "","validation_eer":f"{uthr['eer']:.12g}" if threshold_eligible else "",
                    "genuine_claims":ug,"impostor_claims":ui,"eligible":int(threshold_eligible)
                })
                pp0 = np.asarray(sigmoid(uraw), dtype=np.float64)
                pp1 = up
                cm0, cm1 = calibration_metrics(uy, pp0, uw), calibration_metrics(uy, pp1, uw)
                calibration_diag.append({
                    "cohort":cohort,"model_family":model,"scope":"user","claimed_user_id":u,"temperature":f"{uT:.12g}",
                    "nll_before":f"{cm0['nll']:.12g}","nll_after":f"{cm1['nll']:.12g}",
                    "brier_before":f"{cm0['brier']:.12g}","brier_after":f"{cm1['brier']:.12g}",
                    "ece10_before":f"{cm0['ece10']:.12g}","ece10_after":f"{cm1['ece10']:.12g}"
                })
            macro_auc = float(np.mean(user_aucs)) if user_aucs else float("nan")
            pooled_auc = weighted_auc(y, raw, global_w)
            eligibility_rows.append({
                "cohort":cohort,"model_family":model,"fit_protocol":FIT_PROTOCOL,"validation_claims":len(rows),
                "users_total":len(set(users)),"users_personal_calibrator_eligible":sum(int(v['eligible']) for v in personal.values()),
                "users_global_fallback":sum(int(not v['eligible']) for v in personal.values()),
                "validation_macro_user_auc_raw":f"{macro_auc:.12g}" if math.isfinite(macro_auc) else "",
                "validation_user_equal_pooled_auc_raw":f"{pooled_auc:.12g}" if math.isfinite(pooled_auc) else "",
                "users_auc_below_0_5":user_auc_lt_half,"score_direction_flipped":0,
                "eligible_for_test_application":1,"test_metrics_computed":0
            })
            if math.isfinite(macro_auc) and macro_auc < 0.5:
                warnings.append({"severity":"WARN","code":"WEAK_VALIDATION_DISCRIMINATION","cohort":cohort,"model_family":model,"claimed_user_id":"ALL","message":f"Macro-user validation AUC={macro_auc:.4f}<0.5. Score direction was not flipped; model retained for unbiased baseline reporting."})
            if gbound:
                warnings.append({"severity":"WARN","code":"GLOBAL_TEMPERATURE_BOUNDARY","cohort":cohort,"model_family":model,"claimed_user_id":"ALL","message":f"Global temperature reached optimization boundary: T={gT:.6g}."})
            policy[(cohort, model)] = {"global_temperature":gT,"global_threshold":gthr["threshold"],"personal":personal}

        # Policy is frozen here. Test files are only transformed; no Test aggregates are computed.
        policy_frozen_at = utc_now()
        policy_json = {
            "stage":"4A","version":VERSION,"policy_frozen_at_utc":policy_frozen_at,
            "fit_protocol":FIT_PROTOCOL,"fit_split":"validation","test_used_for_fitting":False,
            "calibration":"temperature_scaling_sigmoid(raw_score / T)",
            "personal_eligibility":{"min_genuine":PERSONAL_MIN_GENUINE,"min_impostor":PERSONAL_MIN_IMPOSTOR,"min_impostor_identities":PERSONAL_MIN_IMPOSTOR_IDENTITIES},
            "threshold_rule":"EER threshold minimizing |FAR-FRR| on calibrated balanced validation scores",
            "fallback":"global model/cohort temperature and user-equal global EER threshold",
            "score_direction":"declared direction retained; no validation-driven flipping",
        }
        (out / "frozen_stage4a_policy.json").write_text(json.dumps(policy_json, indent=2), encoding="utf-8")

        # Apply frozen policies to every validation/test score file.
        # Compact claim-level schema keeps the review package practical while preserving
        # every field required for authentication evaluation and lineage.
        compact_fields = [
            "claim_id","cohort","protocol","split","source_row_index","window_id","session_id","segment_id",
            "actual_user_id","claimed_user_id","is_genuine","model_id","model_family","raw_score",
            "calibrated_score","calibrator_id","calibration_source","temperature",
            "threshold_id","threshold_source","threshold","prediction"
        ]
        for e in sorted(all_entries, key=lambda x:(x["cohort"],x["model_family"],x["protocol"],x["split"])):
            cohort, model, protocol, split = e["cohort"], e["model_family"], e["protocol"], e["split"]
            rel_out = Path("calibrated_scores") / cohort / model / protocol / f"{split}_claim_scores.csv"
            dst = out / rel_out
            dst.parent.mkdir(parents=True, exist_ok=True)
            src: ZipScoreSource = e["source_obj"]
            name = src.root + e["relative_path"].replace("\\", "/")
            with src.zf.open(name, "r") as fb, dst.open("w", encoding="utf-8", newline="", buffering=1024*1024) as fo:
                text = io.TextIOWrapper(fb, encoding="utf-8-sig", newline="")
                reader = csv.reader(text)
                in_header = next(reader)
                ix = {c:i for i,c in enumerate(in_header)}
                required = ["claim_id","source_row_index","window_id","session_id","segment_id","actual_user_id","claimed_user_id","is_genuine","model_id","model_family","raw_score"]
                missing = [c for c in required if c not in ix]
                if missing:
                    raise RuntimeError(f"Missing columns in {name}: {missing}")
                writer = csv.writer(fo, lineterminator="\n")
                writer.writerow(compact_fields)
                rows_count = finite_count = pred_count = 0
                dup = 0
                seen = set()
                val_y=[]; val_p=[]; val_users=[]; val_pred=[]
                for vals in reader:
                    rows_count += 1
                    cid = vals[ix["claim_id"]]
                    if cid in seen: dup += 1
                    seen.add(cid)
                    u = vals[ix["claimed_user_id"]]
                    pp = policy[(cohort, model)]["personal"][u]
                    raw_text = vals[ix["raw_score"]]
                    raw_s = safe_float(raw_text)
                    cal = float(sigmoid(raw_s / float(pp["temperature"])))
                    th = float(pp["threshold"])
                    pred = int(cal >= th)
                    finite_count += int(math.isfinite(cal))
                    pred_count += pred
                    writer.writerow([
                        cid, cohort, protocol, split, vals[ix["source_row_index"]], vals[ix["window_id"]],
                        vals[ix["session_id"]], vals[ix["segment_id"]], vals[ix["actual_user_id"]], u,
                        vals[ix["is_genuine"]], vals[ix["model_id"]], vals[ix["model_family"]], raw_text,
                        f"{cal:.12g}", pp["calibrator_id"], pp["calibration_source"], f"{float(pp['temperature']):.12g}",
                        pp["threshold_id"], pp["threshold_source"], f"{th:.12g}", pred
                    ])
                    if split == "validation":
                        val_y.append(int(vals[ix["is_genuine"]])); val_p.append(cal); val_users.append(u); val_pred.append(pred)
            if rows_count != int(e["claims"]):
                failures.append({"severity":"FAIL","code":"CLAIM_COUNT_MISMATCH","cohort":cohort,"model_family":model,"claimed_user_id":"","message":f"{protocol}/{split}: expected {e['claims']}, wrote {rows_count}"})
            if finite_count != rows_count or dup:
                failures.append({"severity":"FAIL","code":"CALIBRATED_SCORE_INTEGRITY","cohort":cohort,"model_family":model,"claimed_user_id":"","message":f"{protocol}/{split}: finite={finite_count}/{rows_count}, duplicate_claim_ids={dup}"})
            score_index_out.append({
                "cohort":cohort,"model_family":model,"protocol":protocol,"split":split,"relative_path":rel_out.as_posix(),
                "claims":rows_count,"finite_calibrated_scores":finite_count,"duplicate_claim_ids":dup,"predicted_genuine":pred_count,
                "sha256":sha256_file(dst),"test_metrics_computed":0
            })
            integrity.append({"check":"score_file_claim_count","cohort":cohort,"model_family":model,"protocol":protocol,"split":split,"status":"PASS" if rows_count==int(e['claims']) else "FAIL","observed":rows_count,"expected":e['claims']})
            integrity.append({"check":"finite_calibrated_scores","cohort":cohort,"model_family":model,"protocol":protocol,"split":split,"status":"PASS" if finite_count==rows_count else "FAIL","observed":finite_count,"expected":rows_count})
            integrity.append({"check":"duplicate_claim_ids","cohort":cohort,"model_family":model,"protocol":protocol,"split":split,"status":"PASS" if dup==0 else "FAIL","observed":dup,"expected":0})
            if split == "validation":
                vy=np.array(val_y,dtype=np.int8); vp=np.array(val_p,dtype=np.float64); vpred=np.array(val_pred,dtype=np.int8)
                vw=make_user_equal_class_weights(val_users,vy)
                bm=binary_metrics(vy,vpred,vp,vw)
                threshold_diag.append({
                    "cohort":cohort,"model_family":model,"protocol":protocol,"scope":"user_equal_global","claimed_user_id":"ALL",
                    "claims":len(vy),"far":f"{bm['far']:.12g}","frr":f"{bm['frr']:.12g}","auc":f"{bm['auc']:.12g}",
                    "accuracy":f"{bm['accuracy']:.12g}","f1":f"{bm['f1']:.12g}","mcc":f"{bm['mcc']:.12g}","test_metrics_computed":0
                })
                val_users_a=np.array(val_users,dtype=object)
                for u in sorted(set(val_users), key=lambda x:int(x.replace('User',''))):
                    m=val_users_a==u
                    uy=vy[m]; up=vp[m]; upr=vpred[m]; uwu=make_class_balanced_weights(uy)
                    bm=binary_metrics(uy,upr,up,uwu)
                    threshold_diag.append({
                        "cohort":cohort,"model_family":model,"protocol":protocol,"scope":"user","claimed_user_id":u,
                        "claims":int(m.sum()),"far":f"{bm['far']:.12g}","frr":f"{bm['frr']:.12g}","auc":f"{bm['auc']:.12g}" if math.isfinite(bm['auc']) else "",
                        "accuracy":f"{bm['accuracy']:.12g}","f1":f"{bm['f1']:.12g}","mcc":f"{bm['mcc']:.12g}","test_metrics_computed":0
                    })

        write_csv(out/"calibrator_parameters.csv", calibrator_rows, ["cohort","model_family","scope","claimed_user_id","calibrator_id","fit_protocol","fit_split","temperature","genuine_claims","impostor_claims","impostor_identities","eligible","nll_before","nll_after","temperature_at_search_boundary"])
        write_csv(out/"threshold_policy.csv", threshold_rows, ["cohort","model_family","scope","claimed_user_id","threshold_id","threshold_source","fit_protocol","fit_split","threshold","validation_far","validation_frr","validation_eer","genuine_claims","impostor_claims","eligible"])
        write_csv(out/"model_eligibility_audit.csv", eligibility_rows, ["cohort","model_family","fit_protocol","validation_claims","users_total","users_personal_calibrator_eligible","users_global_fallback","validation_macro_user_auc_raw","validation_user_equal_pooled_auc_raw","users_auc_below_0_5","score_direction_flipped","eligible_for_test_application","test_metrics_computed"])
        write_csv(out/"validation_calibration_diagnostics.csv", calibration_diag, ["cohort","model_family","scope","claimed_user_id","temperature","nll_before","nll_after","brier_before","brier_after","ece10_before","ece10_after"])
        write_csv(out/"validation_threshold_diagnostics.csv", threshold_diag, ["cohort","model_family","protocol","scope","claimed_user_id","claims","far","frr","auc","accuracy","f1","mcc","test_metrics_computed"])
        write_csv(out/"calibrated_score_file_index.csv", score_index_out, ["cohort","model_family","protocol","split","relative_path","claims","finite_calibrated_scores","duplicate_claim_ids","predicted_genuine","sha256","test_metrics_computed"])
        write_csv(out/"integrity_checks.csv", integrity, ["check","cohort","model_family","protocol","split","status","observed","expected"])
        write_csv(out/"warnings_and_failures.csv", warnings+failures, ["severity","code","cohort","model_family","claimed_user_id","message"])

        test_guard = [
            {"guard":"test_used_to_fit_calibrator","value":0,"status":"PASS"},
            {"guard":"test_used_to_select_threshold","value":0,"status":"PASS"},
            {"guard":"test_aggregate_metrics_computed","value":0,"status":"PASS"},
            {"guard":"test_claim_level_transformation_only","value":1,"status":"PASS"},
            {"guard":"score_direction_flips_applied","value":0,"status":"PASS"},
        ]
        write_csv(out/"test_metric_guard.csv", test_guard, ["guard","value","status"])

        summary = {
            "stage":"4A","version":VERSION,"status":"PASS" if not failures else "FAIL",
            "input_stage3b":str(s3b),"input_stage3c":str(s3c),
            "input_stage3b_sha256":sha256_file(s3b),"input_stage3c_sha256":sha256_file(s3c),
            "models_total":len(models),"score_files_transformed":len(score_index_out),
            "claim_scores_transformed":sum(int(r["claims"]) for r in score_index_out),
            "personal_calibrators_eligible":sum(int(r["eligible"]) for r in calibrator_rows if r["scope"]=="user"),
            "personal_calibrator_fallbacks":sum(1-int(r["eligible"]) for r in calibrator_rows if r["scope"]=="user"),
            "warnings":len(warnings),"failures":len(failures),
            "test_metrics_computed":0,"policy_frozen_at_utc":policy_frozen_at,
            "started_at_utc":datetime.fromtimestamp(started,timezone.utc).isoformat(),"finished_at_utc":utc_now(),
            "elapsed_seconds":round(time.time()-started,3)
        }
        (out/"stage4a_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        decisions = f"""# Stage 4A Frozen Decisions\n\n- Version: {VERSION}\n- Calibration fit source: Balanced Validation only.\n- Calibration transform: sigmoid(raw_score / T).\n- Personal calibration eligibility: >= {PERSONAL_MIN_GENUINE} genuine, >= {PERSONAL_MIN_IMPOSTOR} impostor, >= {PERSONAL_MIN_IMPOSTOR_IDENTITIES} impostor identities.\n- Fallback: global cohort/model temperature fitted with user-equal, class-balanced weights.\n- Threshold: EER operating point selected on calibrated Balanced Validation only.\n- Personal threshold uses the same eligibility rule; otherwise global user-equal EER fallback.\n- Declared score direction is retained. No validation-driven sign flipping.\n- Validation diagnostics may be computed. Test aggregate metrics are prohibited in Stage 4A.\n- Test claims are transformed only after the policy is frozen.\n- Models with weak validation discrimination remain eligible as transparent baselines and are flagged, not silently removed.\n"""
        (out/"stage4a_frozen_decisions.md").write_text(decisions,encoding="utf-8")

        # Manifest excludes itself and completion marker.
        manifest_rows=[]
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.name not in {"output_manifest_sha256.csv","STAGE4A_COMPLETE.ok"}:
                manifest_rows.append({"relative_path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":sha256_file(p)})
        write_csv(out/"output_manifest_sha256.csv",manifest_rows,["relative_path","size_bytes","sha256"])
        complete = {"stage":"4A","version":VERSION,"status":summary["status"],"manifest_files":len(manifest_rows),"zip_crc_verified":True,"test_metrics_computed":0}
        (out/"STAGE4A_COMPLETE.ok").write_text(json.dumps(complete,indent=2),encoding="utf-8")
        create_zip(out,out_zip)
        print(json.dumps({**summary,"manifest_files_verified":len(manifest_rows),"zip_crc_verified":True,"output_zip":str(out_zip)},indent=2))
        return 0 if not failures else 2
    finally:
        for s in sources:
            s.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

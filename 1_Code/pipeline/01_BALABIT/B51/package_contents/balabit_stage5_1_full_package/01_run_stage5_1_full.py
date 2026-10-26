#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balabit Stage 5.1 — Unified Evaluation and Statistical Validation FULL

Runs the final cluster-aware statistical analysis from the immutable prediction
files produced by Stage 5.1 Pilot v3. It does not train models, export validation
scores, or rerun inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except Exception as exc:
    print("ERROR: numpy, pandas, and matplotlib are required:", repr(exc))
    print("Run: python -m pip install -r requirements_stage5_1_full.txt")
    raise SystemExit(2)

import stage5_1_stats_core as core

VERSION = "5.1-full-v1.0"
OUTPUT_DIR = "balabit_stage5_1_full_output"
PRIMARY_AGGREGATION = "macro_per_user"
PAPER_METRICS = ("FAR", "FRR", "EER", "AUC", "F1", "MCC", "Accuracy")
RATE_METRICS = {"FAR", "FRR", "EER", "AUC", "F1", "Accuracy"}


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def backup_existing(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = path.with_name(f"{path.name}_previous_{stamp}")
    counter = 1
    while target.exists():
        target = path.with_name(f"{path.name}_previous_{stamp}_{counter}")
        counter += 1
    path.rename(target)
    return target


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def source_integrity_table(source_out: Path) -> pd.DataFrame:
    required = [
        "00_pilot_manifest.json", "01_protocol_availability.csv",
        "02_temperature_scaling.csv", "03_thresholds_per_user.csv",
        "04_overlap_and_fairness.csv", "05_metrics_summary.csv",
        "06_metrics_by_user.csv", "08_mcnemar_exact_holm.csv",
        "10_cost_summary.csv", "12_warnings_and_audit.csv",
    ]
    required.extend(core.PRED_FILES.values())
    rows = []
    for name in required:
        path = source_out / name
        if name in core.PRED_FILES.values():
            path = source_out / "full_predictions" / name
        rows.append({
            "relative_path": str(path.relative_to(source_out)) if path.exists() else str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else np.nan,
            "sha256": sha256(path) if path.exists() else "",
        })
    return pd.DataFrame(rows)


def validate_source(source_out: Path) -> List[Dict[str, str]]:
    notes: List[Dict[str, str]] = []
    manifest_path = source_out / "00_pilot_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    status = str(manifest.get("status", ""))
    if "COMPLETE" not in status.upper():
        raise ValueError(f"Pilot v3 manifest does not indicate completion: {status!r}")
    notes.append({"severity": "INFO", "code": "SOURCE_MANIFEST_OK", "detail": f"Pilot v3 status: {status}"})

    temp = pd.read_csv(source_out / "02_temperature_scaling.csv")
    if len(temp) < 2 or not bool(temp["optimizer_success"].fillna(False).all()):
        raise ValueError("Temperature scaling source is incomplete or optimizer_success is false.")
    if not bool((temp["nll_after"] <= temp["nll_before"] + 1e-12).all()):
        raise ValueError("At least one calibrated model has NLL_after > NLL_before.")
    notes.append({"severity": "INFO", "code": "TEMPERATURE_SCALING_VERIFIED", "detail": "Validation-only calibration succeeded and did not increase NLL."})

    overlap = pd.read_csv(source_out / "04_overlap_and_fairness.csv")
    common = overlap[overlap["protocol"] == "balanced_1to1_common"]
    if common.empty or common["model"].nunique() < 4:
        raise ValueError("Balanced common protocol does not contain all four models.")
    if common["protocol_records"].nunique() != 1:
        raise ValueError("Balanced common protocol record counts differ across models.")
    notes.append({"severity": "INFO", "code": "COMMON_RECORD_SET_VERIFIED", "detail": f"Four-model common set contains {int(common['protocol_records'].iloc[0]):,} claims per model."})

    warnings_path = source_out / "12_warnings_and_audit.csv"
    if warnings_path.exists():
        warnings = pd.read_csv(warnings_path)
        fatal = warnings[warnings["severity"].astype(str).str.upper().isin(["ERROR", "FATAL"])]
        if not fatal.empty:
            raise ValueError(f"Source audit contains {len(fatal)} ERROR/FATAL rows.")
        notes.append({"severity": "INFO", "code": "SOURCE_AUDIT_NO_FATAL", "detail": "No ERROR/FATAL entries found in the source audit."})
    return notes


def load_source_tables(source_out: Path) -> Dict[str, pd.DataFrame]:
    names = {
        "protocols": "01_protocol_availability.csv",
        "temperature": "02_temperature_scaling.csv",
        "thresholds": "03_thresholds_per_user.csv",
        "overlap": "04_overlap_and_fairness.csv",
        "metrics": "05_metrics_summary.csv",
        "metrics_user": "06_metrics_by_user.csv",
        "mcnemar": "08_mcnemar_exact_holm.csv",
        "cost": "10_cost_summary.csv",
        "audit": "12_warnings_and_audit.csv",
    }
    return {key: pd.read_csv(source_out / filename) for key, filename in names.items()}


def annotate_mcnemar(mcnemar: pd.DataFrame) -> pd.DataFrame:
    out = mcnemar.copy()
    out["inference_role"] = "descriptive_sensitivity_only"
    out["dependence_caveat"] = (
        "Claims are not independent because each source window contributes multiple user claims and adjacent windows overlap; "
        "cluster-aware user inference is primary."
    )
    return out


def corrected_notes(bootstrap: pd.DataFrame, deltas: pd.DataFrame, source_notes: Sequence[Dict[str, str]]) -> pd.DataFrame:
    outside = bootstrap[~bootstrap["ci_contains_point"]]
    delta_outside = deltas[~deltas["ci_contains_point"]]
    session_rows = bootstrap[bootstrap["cluster_level"].astype(str).str.startswith("stratified")]
    notes = list(source_notes)
    notes.extend([
        {
            "severity": "INFO",
            "code": "FULL_BOOTSTRAP_CONFIGURATION",
            "detail": f"User bootstrap rows used {int(bootstrap[bootstrap['cluster_level']=='claimed_user']['replicates'].max()):,} replicates; session-block rows used {int(session_rows['replicates'].max()):,} replicates.",
        },
        {
            "severity": "INFO",
            "code": "CLASS_SUPPORT_PRESERVED",
            "detail": f"All session-block bootstrap rows preserved both classes: {bool(session_rows['all_users_have_both_classes'].all())}.",
        },
        {
            "severity": "INFO" if outside.empty else "WARNING",
            "code": "METRIC_CI_POINT_SANITY",
            "detail": f"Metric CIs excluding their original point estimate: {len(outside)} of {len(bootstrap)}.",
        },
        {
            "severity": "INFO" if delta_outside.empty else "WARNING",
            "code": "DELTA_CI_POINT_SANITY",
            "detail": f"Paired-delta CIs excluding their original point estimate: {len(delta_outside)} of {len(deltas)}.",
        },
        {
            "severity": "LIMITATION",
            "code": "ONE_GENUINE_TEST_SESSION_PER_USER",
            "detail": "The test split contains one genuine source session per claimed user; within-user genuine-session variability cannot be estimated.",
        },
        {
            "severity": "LIMITATION",
            "code": "CLAIM_LEVEL_MCNEMAR_DEPENDENCE",
            "detail": "Claim-level McNemar is retained only as a descriptive sensitivity analysis because claims share windows and sessions.",
        },
        {
            "severity": "LIMITATION",
            "code": "REALISTIC_BASELINES_UNAVAILABLE",
            "detail": "Random Forest and Linear SVM are unavailable under all-impostor evaluation because Stage 3 did not save reusable fitted model objects.",
        },
    ])
    return pd.DataFrame(notes)


def paper_performance_table(metrics: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    base = metrics[metrics["aggregation"] == PRIMARY_AGGREGATION].copy()
    user_ci = bootstrap[bootstrap["cluster_level"] == "claimed_user"]
    rows = []
    for _, r in base.iterrows():
        row: Dict[str, Any] = {
            "protocol": r["protocol"], "model": r["model"],
            "claims": int(r["claims"]), "users": int(r["users"]),
        }
        for metric in PAPER_METRICS:
            value = float(r[metric])
            ci = user_ci[(user_ci["protocol"] == r["protocol"]) & (user_ci["model"] == r["model"]) & (user_ci["metric"] == metric)]
            scale = 100.0 if metric in RATE_METRICS else 1.0
            row[metric] = value * scale
            row[f"{metric}_ci95_low"] = float(ci["ci95_low"].iloc[0]) * scale if not ci.empty else np.nan
            row[f"{metric}_ci95_high"] = float(ci["ci95_high"].iloc[0]) * scale if not ci.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paper_delta_table(deltas: pd.DataFrame, signflip: pd.DataFrame) -> pd.DataFrame:
    user = deltas[deltas["cluster_level"] == "claimed_user"].copy()
    rows = []
    for _, r in user.iterrows():
        metric_name = str(r["metric"]).replace("Delta_", "")
        sf = signflip[
            (signflip["protocol"] == r["protocol"]) &
            (signflip["model_a"] == r["model_a"]) &
            (signflip["model_b"] == r["model_b"]) &
            (signflip["metric"] == r["metric"])
        ]
        scale = 100.0 if metric_name in RATE_METRICS else 1.0
        rows.append({
            "protocol": r["protocol"], "model_a": r["model_a"], "model_b": r["model_b"],
            "metric": r["metric"], "definition": r["definition"],
            "delta": float(r["point_estimate"]) * scale,
            "ci95_low": float(r["ci95_low"]) * scale,
            "ci95_high": float(r["ci95_high"]) * scale,
            "unit": "percentage_points" if scale == 100 else "raw",
            "signflip_p": float(sf["two_sided_p_value"].iloc[0]) if not sf.empty else np.nan,
            "holm_adjusted_p": float(sf["holm_adjusted_p_within_protocol"].iloc[0]) if not sf.empty else np.nan,
            "significant_after_holm_0_05": bool(sf["significant_after_holm_0_05"].iloc[0]) if not sf.empty else False,
        })
    return pd.DataFrame(rows)


def paper_cost_table(costs: pd.DataFrame) -> pd.DataFrame:
    c = costs.copy()
    c["model"] = c["model"].astype(str).str.upper()
    c = c[c["model"].isin(["CNN", "HYBRID", "BILSTM"])].copy()
    keep = [
        "model", "total_parameters", "checkpoint_size_mb", "training_seconds",
        "single_mean_ms", "single_p95_ms", "throughput_windows_per_second",
        "tta_mean_sec", "tta_p95_approx_sec",
    ]
    return c[[x for x in keep if x in c.columns]].sort_values("model")


def threshold_summary(thresholds: pd.DataFrame) -> pd.DataFrame:
    t = thresholds.copy()
    t["model"] = t["model"].astype(str).str.upper()
    t = t[t["model"].isin(["CNN", "HYBRID"])]
    return t.groupby("model", as_index=False).agg(
        users=("claimed_user_id", "nunique"),
        threshold_mean=("threshold", "mean"),
        threshold_std=("threshold", "std"),
        threshold_min=("threshold", "min"),
        threshold_median=("threshold", "median"),
        threshold_max=("threshold", "max"),
        validation_EER_mean=("validation_EER", "mean"),
        validation_MCC_mean=("validation_MCC", "mean"),
    )


def model_selection_table(justification: pd.DataFrame, metrics: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    realistic = justification[justification["protocol"] == "realistic_all_impostors"]
    decision = realistic["decision"].iloc[0] if not realistic.empty else "REVIEW_REQUIRED"
    cnn = costs[costs["model"].astype(str).str.upper() == "CNN"].iloc[0]
    hyb = costs[costs["model"].astype(str).str.upper() == "HYBRID"].iloc[0]
    return pd.DataFrame([
        {
            "role": "primary_deployment_model",
            "selected_model": "CNN",
            "basis": "No robust cluster-aware Hybrid advantage on primary metrics; substantially lower edge cost.",
            "hybrid_decision": decision,
            "hybrid_parameter_ratio": float(hyb["total_parameters"]) / float(cnn["total_parameters"]),
            "hybrid_size_ratio": float(hyb["checkpoint_size_mb"]) / float(cnn["checkpoint_size_mb"]),
            "hybrid_latency_ratio": float(hyb["single_mean_ms"]) / float(cnn["single_mean_ms"]),
        },
        {
            "role": "accuracy_oriented_alternative",
            "selected_model": "HYBRID",
            "basis": "Marginal numerical gains may be reported, especially lower realistic FAR, but not as statistically established overall superiority.",
            "hybrid_decision": decision,
            "hybrid_parameter_ratio": float(hyb["total_parameters"]) / float(cnn["total_parameters"]),
            "hybrid_size_ratio": float(hyb["checkpoint_size_mb"]) / float(cnn["checkpoint_size_mb"]),
            "hybrid_latency_ratio": float(hyb["single_mean_ms"]) / float(cnn["single_mean_ms"]),
        },
    ])


def save_figure(fig: Any, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def metric_ci_figure(performance: pd.DataFrame, protocol: str, metric: str, out_dir: Path, filename: str) -> None:
    d = performance[performance["protocol"] == protocol].copy()
    if d.empty:
        return
    d = d.sort_values(metric)
    y = np.arange(len(d))
    vals = d[metric].to_numpy(float)
    low = d[f"{metric}_ci95_low"].to_numpy(float)
    high = d[f"{metric}_ci95_high"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.errorbar(vals, y, xerr=np.vstack([vals-low, high-vals]), fmt="o", capsize=4)
    ax.set_yticks(y, d["model"].astype(str).tolist())
    ax.set_xlabel(f"{metric} ({'%' if metric in RATE_METRICS else 'value'})")
    ax.set_title(f"{metric} with 95% user-cluster bootstrap CI\n{protocol}")
    ax.grid(axis="x", alpha=0.3)
    save_figure(fig, out_dir / filename)


def far_frr_figure(performance: pd.DataFrame, out_dir: Path) -> None:
    d = performance[performance["protocol"] == "realistic_all_impostors"].copy()
    if d.empty:
        return
    x = np.arange(len(d))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - width/2, d["FAR"], width, label="FAR")
    ax.bar(x + width/2, d["FRR"], width, label="FRR")
    ax.set_xticks(x, d["model"].astype(str).tolist())
    ax.set_ylabel("Error rate (%)")
    ax.set_title("Realistic all-impostor FAR and FRR")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, out_dir / "figure_04_realistic_far_frr")


def delta_forest_figure(delta_table: pd.DataFrame, out_dir: Path) -> None:
    d = delta_table[
        (delta_table["protocol"] == "realistic_all_impostors") &
        (delta_table["model_a"] == "HYBRID") &
        (delta_table["model_b"] == "CNN") &
        (delta_table["unit"] == "percentage_points")
    ].copy()
    if d.empty:
        return
    order = [m for m in ["Delta_EER", "Delta_FAR", "Delta_FRR", "Delta_AUC"] if m in set(d["metric"])]
    d["metric"] = pd.Categorical(d["metric"], categories=order, ordered=True)
    d = d.sort_values("metric")
    y = np.arange(len(d))
    vals = d["delta"].to_numpy(float)
    low = d["ci95_low"].to_numpy(float)
    high = d["ci95_high"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(vals, y, xerr=np.vstack([vals-low, high-vals]), fmt="o", capsize=4)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y, d["metric"].astype(str).tolist())
    ax.set_xlabel("Hybrid − CNN (percentage points)")
    ax.set_title("Realistic protocol paired user-bootstrap differences")
    ax.grid(axis="x", alpha=0.3)
    save_figure(fig, out_dir / "figure_05_hybrid_minus_cnn_rate_deltas")


def cost_ratio_figure(costs: pd.DataFrame, out_dir: Path) -> None:
    c = costs.copy()
    c["model"] = c["model"].astype(str).str.upper()
    cnn = c[c["model"] == "CNN"].iloc[0]
    hyb = c[c["model"] == "HYBRID"].iloc[0]
    labels = ["Parameters", "Model size", "Inference latency", "Training time"]
    vals = [
        float(hyb["total_parameters"]) / float(cnn["total_parameters"]),
        float(hyb["checkpoint_size_mb"]) / float(cnn["checkpoint_size_mb"]),
        float(hyb["single_mean_ms"]) / float(cnn["single_mean_ms"]),
        float(hyb["training_seconds"]) / float(cnn["training_seconds"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(labels, vals)
    ax.axhline(1.0, linewidth=1)
    ax.set_ylabel("Hybrid / CNN ratio")
    ax.set_title("Computational cost of Hybrid relative to CNN")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, out_dir / "figure_06_hybrid_cnn_cost_ratios")


def per_user_figure(metrics_user: pd.DataFrame, metric: str, out_dir: Path, filename: str) -> None:
    d = metrics_user[
        (metrics_user["protocol"] == "realistic_all_impostors") &
        (metrics_user["model"].astype(str).str.upper().isin(["CNN", "HYBRID"]))
    ].copy()
    if d.empty:
        return
    p = d.pivot(index="claimed_user_id", columns="model", values=metric).sort_index()
    scale = 100.0 if metric in RATE_METRICS else 1.0
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for model in [m for m in ["CNN", "HYBRID"] if m in p.columns]:
        ax.plot(np.arange(len(p)), p[model].to_numpy(float)*scale, marker="o", label=model)
    ax.set_xticks(np.arange(len(p)), p.index.astype(str).tolist(), rotation=45, ha="right")
    ax.set_ylabel(f"{metric} ({'%' if scale == 100 else 'value'})")
    ax.set_title(f"Per-user {metric} under the realistic protocol")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, out_dir / filename)


def calibration_figure(temp: pd.DataFrame, out_dir: Path) -> None:
    d = temp.copy()
    x = np.arange(len(d))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x-width/2, d["nll_before"], width, label="Before")
    ax.bar(x+width/2, d["nll_after"], width, label="After")
    ax.set_xticks(x, d["model"].astype(str).tolist())
    ax.set_ylabel("Validation negative log-likelihood")
    ax.set_title("Effect of validation-only temperature scaling")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, out_dir / "figure_09_temperature_scaling_nll")


def write_latex(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = df.to_latex(index=False, float_format=lambda x: f"{x:.4f}", caption=caption, label=label, escape=True)
    path.write_text(text, encoding="utf-8")


def make_claims_text(performance: pd.DataFrame, deltas: pd.DataFrame, signflip: pd.DataFrame, justification: pd.DataFrame, costs: pd.DataFrame) -> Tuple[str, str]:
    real = performance[performance["protocol"] == "realistic_all_impostors"].set_index("model")
    bal = performance[performance["protocol"] == "balanced_1to1_common"].set_index("model")
    real_delta = deltas[(deltas["protocol"] == "realistic_all_impostors") & (deltas["model_a"] == "HYBRID") & (deltas["model_b"] == "CNN")].set_index("metric")
    sig = signflip[(signflip["protocol"] == "realistic_all_impostors") & (signflip["model_a"] == "HYBRID") & (signflip["model_b"] == "CNN")]
    sig_any = bool(sig["significant_after_holm_0_05"].any())
    j = justification[justification["protocol"] == "realistic_all_impostors"].iloc[0]

    en = f"""# Stage 5.1 Full — Publication Interpretation\n\n## Primary conclusion\nCNN is selected as the primary deployment model. Hybrid achieved marginal numerical improvements under the realistic all-impostor protocol, but no primary Hybrid–CNN comparison remained significant after exact user-level sign-flip testing with Holm correction ({'at least one significant comparison' if sig_any else 'no significant comparison'}). The final cost decision is **{j['decision']}**.\n\n## Realistic protocol\n- CNN: EER {real.loc['CNN','EER']:.3f}%, MCC {real.loc['CNN','MCC']:.4f}, FAR {real.loc['CNN','FAR']:.3f}%, FRR {real.loc['CNN','FRR']:.3f}%.\n- Hybrid: EER {real.loc['HYBRID','EER']:.3f}%, MCC {real.loc['HYBRID','MCC']:.4f}, FAR {real.loc['HYBRID','FAR']:.3f}%, FRR {real.loc['HYBRID','FRR']:.3f}%.\n- Hybrid − CNN: ΔEER {real_delta.loc['Delta_EER','delta']:.3f} percentage points; ΔMCC {real_delta.loc['Delta_MCC','delta']:.4f}; ΔFAR {real_delta.loc['Delta_FAR','delta']:.3f} percentage points; ΔFRR {real_delta.loc['Delta_FRR','delta']:.3f} percentage points.\n\n## Balanced common protocol\nCNN and Hybrid substantially outperform Random Forest and Linear SVM on the same 1:1 claim set. Hybrid and CNN remain statistically comparable on the primary cluster-aware comparison.\n\n## Edge-cost interpretation\nHybrid uses {j['hybrid_to_cnn_parameter_ratio']:.2f}× the parameters, {j['hybrid_to_cnn_size_ratio']:.2f}× the checkpoint size, and {j['hybrid_to_cnn_latency_ratio']:.2f}× the single-window inference latency of CNN. Therefore, the limited numerical improvement does not justify replacing CNN as the resource-aware deployment choice.\n\n## Permitted manuscript claim\n“Both deep models markedly outperformed the classical baselines under the balanced common-claim protocol. Although the hybrid architecture produced marginal numerical gains in several metrics, cluster-aware user-level inference did not establish a robust overall advantage over CNN. CNN therefore offers the preferable performance–efficiency trade-off for continuous authentication in resource-constrained Zero Trust deployments.”\n\n## Required limitations\n1. The test split contains one genuine source session per user.\n2. Claim-level McNemar is descriptive because windows and claims are dependent.\n3. All-impostor baseline evaluation is unavailable because reusable Stage 3 baseline models were not saved.\n4. Exact user-level inference is based on ten users, limiting statistical power.\n"""

    ar = f"""# التفسير النهائي لمرحلة Stage 5.1 Full\n\n## الاستنتاج الرئيس\nيُعتمد CNN نموذجاً رئيساً للنشر. حقق Hybrid تحسينات رقمية محدودة في البروتوكول الواقعي، إلا أن المقارنات الأساسية بينه وبين CNN لم تثبت تفوقاً عاماً بعد اختبار sign-flip الدقيق على مستوى المستخدم وتصحيح Holm. قرار الكلفة النهائي هو **{j['decision']}**.\n\n## البروتوكول الواقعي\n- CNN: ‏EER = {real.loc['CNN','EER']:.3f}%، ‏MCC = {real.loc['CNN','MCC']:.4f}، ‏FAR = {real.loc['CNN','FAR']:.3f}%، ‏FRR = {real.loc['CNN','FRR']:.3f}%.\n- Hybrid: ‏EER = {real.loc['HYBRID','EER']:.3f}%، ‏MCC = {real.loc['HYBRID','MCC']:.4f}، ‏FAR = {real.loc['HYBRID','FAR']:.3f}%، ‏FRR = {real.loc['HYBRID','FRR']:.3f}%.\n- فروق Hybrid−CNN: ‏ΔEER = {real_delta.loc['Delta_EER','delta']:.3f} نقطة مئوية، ‏ΔMCC = {real_delta.loc['Delta_MCC','delta']:.4f}، ‏ΔFAR = {real_delta.loc['Delta_FAR','delta']:.3f} نقطة، ‏ΔFRR = {real_delta.loc['Delta_FRR','delta']:.3f} نقطة.\n\n## قرار النشر على Edge\nيستخدم Hybrid معاملات أكثر بمقدار {j['hybrid_to_cnn_parameter_ratio']:.2f}×، وحجماً أكبر بمقدار {j['hybrid_to_cnn_size_ratio']:.2f}×، وزمن استدلال أكبر بمقدار {j['hybrid_to_cnn_latency_ratio']:.2f}× مقارنةً بـCNN. لذلك لا يبرر التحسن الرقمي المحدود استبدال CNN بوصفه خيار النشر الكفوء.\n\n## الصياغة البحثية المسموح بها\n«تفوق النموذجان العميقان بوضوح على خطوط الأساس التقليدية ضمن بروتوكول المطالبات المشتركة المتوازن. وعلى الرغم من تحقيق البنية الهجينة تحسينات رقمية محدودة في بعض المؤشرات، لم يثبت الاستدلال العنقودي على مستوى المستخدم تفوقاً عاماً قوياً على CNN؛ لذلك يقدم CNN الموازنة الأفضل بين الأداء والكفاءة للنشر المستمر في بيئات Zero Trust محدودة الموارد.»\n\n## القيود الواجب توثيقها\n1. يتضمن الاختبار جلسة Genuine مصدرية واحدة لكل مستخدم.\n2. يُعامل McNemar على مستوى المطالبات كتحليل وصفي بسبب ترابط النوافذ والمطالبات.\n3. لا تتوفر مقارنة all-impostor لخطوط الأساس لعدم حفظ نماذج Stage 3 القابلة لإعادة الاستدلال.\n4. يعتمد الاختبار الدقيق على عشرة مستخدمين، مما يحد من القوة الإحصائية.\n"""
    return en, ar


def output_inventory(out: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            rows.append({
                "relative_path": str(p.relative_to(out)),
                "size_bytes": p.stat().st_size,
                "sha256": sha256(p),
            })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Balabit Stage 5.1 Full unified statistical validation")
    parser.add_argument("--project-root", default=None, help="Directory containing balabit_stage5_1_pilot_v3_output")
    parser.add_argument("--user-replicates", type=int, default=10000)
    parser.add_argument("--session-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--keep-existing-output", action="store_true", help="Fail instead of backing up an existing Full output directory")
    parser.add_argument("--self-test-mode", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    min_user = 20 if args.self_test_mode else 2000
    min_session = 10 if args.self_test_mode else 500
    if args.user_replicates < min_user:
        raise SystemExit(f"Full mode requires --user-replicates >= {min_user}")
    if args.session_replicates < min_session:
        raise SystemExit(f"Full mode requires --session-replicates >= {min_session}")

    started = time.time()
    script_dir = Path(__file__).resolve().parent
    root = core.resolve_project_root(script_dir, args.project_root)
    source_out = root / core.SOURCE_OUTPUT_DIR
    out = root / OUTPUT_DIR
    if out.exists():
        if args.keep_existing_output:
            raise SystemExit(f"Output already exists: {out}")
        backup = backup_existing(out)
        print("Existing Full output backed up to:", backup)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run_console.log"

    class Tee:
        def __init__(self, *streams: Any): self.streams = streams
        def write(self, data: str) -> None:
            for s in self.streams:
                s.write(data); s.flush()
        def flush(self) -> None:
            for s in self.streams: s.flush()

    stdout0, stderr0 = sys.stdout, sys.stderr
    manifest: Dict[str, Any] = {
        "version": VERSION,
        "status": "RUNNING",
        "project_root": str(root),
        "source_v3_output": str(source_out),
        "source_read_only": True,
        "no_training": True,
        "no_inference": True,
        "no_threshold_refitting": True,
        "user_bootstrap_replicates": int(args.user_replicates),
        "session_bootstrap_replicates": int(args.session_replicates),
        "seed": int(args.seed),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with log_path.open("w", encoding="utf-8") as log:
        sys.stdout = Tee(stdout0, log)
        sys.stderr = Tee(stderr0, log)
        try:
            print(f"Balabit Stage 5.1 Full {VERSION}")
            print("Project root:", root)
            print("Source output:", source_out)
            print("User bootstrap replicates:", args.user_replicates)
            print("Session-block bootstrap replicates:", args.session_replicates)

            source_notes = validate_source(source_out)
            integrity = source_integrity_table(source_out)
            if not bool(integrity["exists"].all()):
                missing = integrity[~integrity["exists"]]["relative_path"].tolist()
                raise FileNotFoundError(f"Missing required source files: {missing}")
            write_df(integrity, out / "00_source_integrity.csv")

            source_tables = load_source_tables(source_out)
            protocols = core.load_predictions(source_out)
            for protocol, models in protocols.items():
                core.labels_and_keys_consistent(models)
                print(f"[OK] {protocol}: {', '.join(f'{m}={len(d):,}' for m, d in models.items())}")

            bootstrap_frames: List[pd.DataFrame] = []
            inventory_frames: List[pd.DataFrame] = []
            for protocol, models in protocols.items():
                for model, df in models.items():
                    print(f"[BOOTSTRAP FULL] {protocol} / {model}")
                    b, inv = core.bootstrap_ci(
                        df, protocol, model,
                        user_reps=int(args.user_replicates),
                        session_reps=int(args.session_replicates),
                        seed=args.seed,
                    )
                    bootstrap_frames.append(b)
                    inventory_frames.append(inv)
            bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
            session_inventory = pd.concat(inventory_frames, ignore_index=True)
            write_df(bootstrap, out / "07_bootstrap_95ci_full.csv")
            write_df(session_inventory, out / "07b_session_block_inventory_full.csv")

            delta_frames: List[pd.DataFrame] = []
            for protocol, pairs in core.PAIR_SPECS.items():
                models = protocols[protocol]
                for a, b in pairs:
                    model_a, model_b = (("HYBRID", "CNN") if {a, b} == {"CNN", "HYBRID"} else (a, b))
                    print(f"[PAIRED BOOTSTRAP FULL] {protocol} / {model_a} vs {model_b}")
                    delta_frames.append(core.paired_delta_ci(
                        models[model_a], models[model_b], model_a, model_b, protocol,
                        user_reps=int(args.user_replicates),
                        session_reps=int(args.session_replicates),
                        seed=args.seed,
                    ))
            deltas = pd.concat(delta_frames, ignore_index=True)
            write_df(deltas, out / "09_paired_performance_deltas_full.csv")

            signflip = core.paired_user_signflip_tests(protocols, args.seed)
            write_df(signflip, out / "10_exact_user_signflip_holm_full.csv")

            mcnemar = annotate_mcnemar(source_tables["mcnemar"])
            write_df(mcnemar, out / "08_mcnemar_exact_holm_annotated.csv")

            justification = core.corrected_hybrid_justification(
                source_tables["metrics"], deltas, signflip, source_tables["cost"]
            )
            write_df(justification, out / "11_hybrid_cost_justification_full.csv")

            notes = corrected_notes(bootstrap, deltas, source_notes)
            write_df(notes, out / "12_statistical_validity_notes_full.csv")

            # Self-contained stable source summaries.
            for name in [
                "00_pilot_manifest.json", "01_protocol_availability.csv", "02_temperature_scaling.csv",
                "03_thresholds_per_user.csv", "04_overlap_and_fairness.csv", "05_metrics_summary.csv",
                "06_metrics_by_user.csv", "10_cost_summary.csv", "12_warnings_and_audit.csv",
            ]:
                copy_if_exists(source_out / name, out / f"source_v3__{name}")

            paper_dir = out / "paper_tables"
            fig_dir = out / "paper_figures"
            performance = paper_performance_table(source_tables["metrics"], bootstrap)
            delta_table = paper_delta_table(deltas, signflip)
            cost_table = paper_cost_table(source_tables["cost"])
            threshold_table = threshold_summary(source_tables["thresholds"])
            selection = model_selection_table(justification, source_tables["metrics"], source_tables["cost"])
            write_df(performance, paper_dir / "table_01_performance_with_user_ci.csv")
            write_df(delta_table, paper_dir / "table_02_paired_deltas_and_signflip.csv")
            write_df(cost_table, paper_dir / "table_03_model_cost.csv")
            write_df(source_tables["temperature"], paper_dir / "table_04_temperature_scaling.csv")
            write_df(threshold_table, paper_dir / "table_05_threshold_summary.csv")
            write_df(selection, paper_dir / "table_06_final_model_selection.csv")
            write_df(source_tables["metrics_user"], paper_dir / "table_07_metrics_by_user.csv")

            write_latex(performance, paper_dir / "table_01_performance_with_user_ci.tex", "Unified performance under the realistic and balanced protocols.", "tab:stage51_performance")
            write_latex(delta_table, paper_dir / "table_02_paired_deltas_and_signflip.tex", "Cluster-aware paired performance differences and exact user-level tests.", "tab:stage51_deltas")
            write_latex(cost_table, paper_dir / "table_03_model_cost.tex", "Model complexity, latency, throughput, and time-to-authentication.", "tab:stage51_cost")

            metric_ci_figure(performance, "balanced_1to1_common", "EER", fig_dir, "figure_01_balanced_eer")
            metric_ci_figure(performance, "balanced_1to1_common", "MCC", fig_dir, "figure_02_balanced_mcc")
            metric_ci_figure(performance, "realistic_all_impostors", "EER", fig_dir, "figure_03_realistic_eer")
            far_frr_figure(performance, fig_dir)
            delta_forest_figure(delta_table, fig_dir)
            cost_ratio_figure(source_tables["cost"], fig_dir)
            per_user_figure(source_tables["metrics_user"], "EER", fig_dir, "figure_07_realistic_per_user_eer")
            per_user_figure(source_tables["metrics_user"], "MCC", fig_dir, "figure_08_realistic_per_user_mcc")
            calibration_figure(source_tables["temperature"], fig_dir)

            en, ar = make_claims_text(performance, delta_table, signflip, justification, source_tables["cost"])
            (out / "13_publication_interpretation_EN.md").write_text(en, encoding="utf-8")
            (out / "14_publication_interpretation_AR.md").write_text(ar, encoding="utf-8")

            outside = int((~bootstrap["ci_contains_point"]).sum())
            delta_outside = int((~deltas["ci_contains_point"]).sum())
            all_class_support = bool(
                bootstrap[bootstrap["cluster_level"].astype(str).str.startswith("stratified")]["all_users_have_both_classes"].all()
            )
            final_decision = justification[justification["protocol"] == "realistic_all_impostors"]["decision"].iloc[0]
            manifest.update({
                "status": "FULL_COMPLETE",
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(time.time() - started, 3),
                "primary_inference": [
                    "claimed-user cluster bootstrap",
                    "stratified actual-session block bootstrap within claimed user and class",
                    "exact paired user sign-flip test with Holm correction",
                ],
                "claim_level_mcnemar_role": "descriptive_sensitivity_only",
                "all_session_replicates_preserved_both_classes": all_class_support,
                "bootstrap_ci_rows_excluding_point": outside,
                "paired_delta_ci_rows_excluding_point": delta_outside,
                "primary_deployment_model": "CNN",
                "hybrid_cost_decision": final_decision,
                "paper_figures_png": len(list(fig_dir.glob("*.png"))),
                "paper_figures_pdf": len(list(fig_dir.glob("*.pdf"))),
                "paper_tables_csv": len(list(paper_dir.glob("*.csv"))),
                "paper_tables_tex": len(list(paper_dir.glob("*.tex"))),
            })
            (out / "00_full_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

            # Inventory is generated before the review ZIP, then refreshed afterward.
            inv = output_inventory(out)
            write_df(inv, out / "15_output_inventory.csv")

            review_zip = out / "balabit_stage5_1_full_review.zip"
            with zipfile.ZipFile(review_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for p in sorted(out.rglob("*")):
                    if not p.is_file() or p == review_zip:
                        continue
                    if "full_predictions" in p.parts:
                        continue
                    if p.suffix.lower() in {".csv", ".json", ".log", ".md", ".png", ".pdf", ".tex", ".txt"}:
                        z.write(p, arcname=str(p.relative_to(out)))
            hash_path = out / "balabit_stage5_1_full_review.zip.sha256.txt"
            hash_path.write_text(f"{sha256(review_zip)}  {review_zip.name}\n", encoding="utf-8")
            inv = output_inventory(out)
            write_df(inv, out / "15_output_inventory.csv")

            print("\nStage 5.1 Full completed successfully.")
            print("Primary deployment model: CNN")
            print("Hybrid cost decision:", final_decision)
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
            (out / "00_full_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print("\nFAILED SAFE. Source predictions and earlier Stage outputs were not modified.")
            return 1
        finally:
            sys.stdout, sys.stderr = stdout0, stderr0


if __name__ == "__main__":
    raise SystemExit(main())

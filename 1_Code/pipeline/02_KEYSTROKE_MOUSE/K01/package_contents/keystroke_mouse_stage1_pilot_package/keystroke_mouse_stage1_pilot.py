#!/usr/bin/env python3
"""
Keystroke–Mouse Stage 1 — Dataset Inventory, Integrity and Leakage Audit

Read-only pilot for ZIP files or directories. It does not modify source files,
does not train models, and processes delimited text files row-by-row.

Python: 3.12+
Dependencies: standard library only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import traceback
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, Iterator, TextIO

VERSION = "1.0.0"
ID_RE = re.compile(r"(^|_)(id|uuid|guid|subject|user|session|window|claim)(_|$)", re.I)
TIME_RE = re.compile(r"time|date|timestamp", re.I)
LABEL_RE = re.compile(r"label|class|target|attack|genuine|impostor|fraud|anomaly", re.I)
USER_RE = re.compile(r"(^|_)(user|subject|participant|person)(_|$)", re.I)
SESSION_RE = re.compile(r"session", re.I)
EVENT_RE = re.compile(r"event|action|type", re.I)
KEY_RE = re.compile(r"key|scan|code", re.I)
MOUSE_RE = re.compile(r"mouse|cursor|xpos|ypos|click|wheel", re.I)
NA_TOKENS = {"", "na", "n/a", "null", "none", "nan", "missing"}
TEXT_EXTS = {".csv", ".tsv", ".txt"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{n} B"


def parse_datetime(value: str) -> datetime | None:
    s = value.strip()
    if not s:
        return None
    candidates = [s, s.replace("Z", "+00:00")]
    for candidate in candidates:
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    formats = (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    )
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def infer_delimiter(sample: str) -> str:
    lines = [ln for ln in sample.splitlines() if ln.strip()][:20]
    if not lines:
        return ","
    counts = {d: sum(line.count(d) for line in lines) for d in ["\t", ",", ";", "|"]}
    # Tabs are highly reliable and avoid commas embedded in user-agent strings.
    if counts["\t"] > 0 and counts["\t"] >= max(counts[","], counts[";"], counts["|"]):
        return "\t"
    try:
        return csv.Sniffer().sniff("\n".join(lines), delimiters=",;\t|").delimiter
    except csv.Error:
        return max(counts, key=counts.get) if max(counts.values()) else ","


def sanitize(value: str, column: str, salt: str) -> str:
    v = value.strip()
    c = column.lower()
    if not v:
        return v
    if ID_RE.search(c) or USER_RE.search(c) or SESSION_RE.search(c) or c in {"_id", "ua", "user_agent"}:
        digest = hashlib.sha256((salt + "|" + v).encode("utf-8", "replace")).hexdigest()[:10]
        prefix = "anon"
        if USER_RE.search(c):
            prefix = "user"
        elif SESSION_RE.search(c):
            prefix = "session"
        elif c in {"ua", "user_agent"}:
            prefix = "ua"
        return f"{prefix}_{digest}"
    if len(v) > 100:
        return v[:97] + "..."
    return v


def looks_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def looks_float(s: str) -> bool:
    try:
        x = float(s)
        return math.isfinite(x)
    except ValueError:
        return False


@dataclass
class ColumnStats:
    name: str
    missing: int = 0
    nonmissing: int = 0
    int_like: int = 0
    float_like: int = 0
    datetime_like: int = 0
    unique_values: set[str] = field(default_factory=set)
    unique_capped: bool = False
    top_values: Counter = field(default_factory=Counter)
    min_number: float | None = None
    max_number: float | None = None
    min_datetime: datetime | None = None
    max_datetime: datetime | None = None

    def update(self, raw: str, unique_cap: int, top_cap: int = 5000) -> None:
        value = (raw or "").strip()
        if value.lower() in NA_TOKENS:
            self.missing += 1
            return
        self.nonmissing += 1
        if not self.unique_capped:
            self.unique_values.add(value)
            if len(self.unique_values) > unique_cap:
                self.unique_values.clear()
                self.unique_capped = True
        if len(self.top_values) < top_cap or value in self.top_values:
            self.top_values[value] += 1
        if looks_int(value):
            self.int_like += 1
        if looks_float(value):
            self.float_like += 1
            num = float(value)
            self.min_number = num if self.min_number is None else min(self.min_number, num)
            self.max_number = num if self.max_number is None else max(self.max_number, num)
        if TIME_RE.search(self.name):
            dt = parse_datetime(value)
            if dt is not None:
                self.datetime_like += 1
                self.min_datetime = dt if self.min_datetime is None else min(self.min_datetime, dt)
                self.max_datetime = dt if self.max_datetime is None else max(self.max_datetime, dt)

    def inferred_type(self) -> str:
        n = max(1, self.nonmissing)
        if self.datetime_like / n >= 0.9:
            return "datetime"
        if self.int_like / n >= 0.99:
            return "integer"
        if self.float_like / n >= 0.99:
            return "float"
        return "string"


@dataclass
class SourceItem:
    container: str
    member: str
    compressed_size: int
    uncompressed_size: int
    opener: callable


class Audit:
    def __init__(self, input_path: Path, output_dir: Path, sample_rows: int, unique_cap: int):
        self.input_path = input_path
        self.output_dir = output_dir
        self.sample_rows = sample_rows
        self.unique_cap = unique_cap
        self.logs: list[str] = []
        self.warnings: list[dict] = []
        self.inventory: list[dict] = []
        self.schemas: dict[str, dict] = {}
        self.samples: list[dict] = []
        self.missing_rows: list[dict] = []
        self.duplicate_rows: list[dict] = []
        self.timestamp_rows: list[dict] = []
        self.label_rows: list[dict] = []
        self.identifier_rows: list[dict] = []
        self.user_session_rows: list[dict] = []
        self.file_summaries: dict[str, dict] = {}
        self.cross_file: dict = {}
        self.salt = sha256_file(input_path)[:16] if input_path.is_file() else hashlib.sha256(str(input_path).encode()).hexdigest()[:16]

    def log(self, msg: str) -> None:
        stamp = datetime.now().isoformat(timespec="seconds")
        self.logs.append(f"[{stamp}] {msg}")

    def warn(self, code: str, severity: str, file: str, message: str) -> None:
        self.warnings.append({"code": code, "severity": severity, "file": file, "message": message})
        self.log(f"WARNING {severity} {code} {file}: {message}")

    def discover(self) -> list[SourceItem]:
        items: list[SourceItem] = []
        p = self.input_path
        if p.is_file() and zipfile.is_zipfile(p):
            zf = zipfile.ZipFile(p)
            bad = zf.testzip()
            if bad:
                self.warn("ZIP_CRC_FAILURE", "critical", p.name, f"Corrupt ZIP member: {bad}")
            for info in zf.infolist():
                if info.is_dir():
                    continue
                ext = PurePosixPath(info.filename).suffix.lower()
                self.inventory.append({
                    "container": p.name,
                    "member": info.filename,
                    "extension": ext,
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                    "compressed_human": human_bytes(info.compress_size),
                    "uncompressed_human": human_bytes(info.file_size),
                    "compression_ratio": round(info.file_size / max(1, info.compress_size), 3),
                    "estimated_dataframe_memory_6x": human_bytes(info.file_size * 6),
                    "status": "candidate" if ext in TEXT_EXTS else "not_parsed",
                })
                if ext in TEXT_EXTS:
                    def make_opener(name=info.filename, archive=p):
                        z = zipfile.ZipFile(archive)
                        raw = z.open(name, "r")
                        # Keep zip handle alive by attaching it to wrapper.
                        raw._parent_zip = z  # type: ignore[attr-defined]
                        return raw
                    items.append(SourceItem(p.name, info.filename, info.compress_size, info.file_size, make_opener))
            self.log(f"Discovered {len(items)} delimited text members in ZIP.")
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                self.inventory.append({
                    "container": str(p),
                    "member": str(f.relative_to(p)),
                    "extension": ext,
                    "compressed_bytes": "",
                    "uncompressed_bytes": f.stat().st_size,
                    "compressed_human": "",
                    "uncompressed_human": human_bytes(f.stat().st_size),
                    "compression_ratio": "",
                    "estimated_dataframe_memory_6x": human_bytes(f.stat().st_size * 6),
                    "status": "candidate" if ext in TEXT_EXTS else "not_parsed",
                })
                if ext in TEXT_EXTS:
                    items.append(SourceItem(str(p), str(f.relative_to(p)), f.stat().st_size, f.stat().st_size, lambda f=f: f.open("rb")))
            self.log(f"Discovered {len(items)} delimited text files in directory.")
        else:
            raise FileNotFoundError(f"Input must be a ZIP file or directory: {p}")
        return items

    @staticmethod
    def open_text(item: SourceItem) -> tuple[TextIO, str, str]:
        raw: BinaryIO = item.opener()
        prefix = raw.read(65536)
        encoding = "utf-8-sig"
        try:
            sample = prefix.decode(encoding)
        except UnicodeDecodeError:
            encoding = "latin-1"
            sample = prefix.decode(encoding, "replace")
        delimiter = infer_delimiter(sample)
        raw.close()
        raw2: BinaryIO = item.opener()
        return io.TextIOWrapper(raw2, encoding=encoding, errors="replace", newline=""), encoding, delimiter

    def audit_file(self, item: SourceItem) -> None:
        key = item.member
        self.log(f"Auditing {key}")
        fh, encoding, delimiter = self.open_text(item)
        try:
            reader = csv.DictReader(fh, delimiter=delimiter)
            if not reader.fieldnames:
                self.warn("NO_HEADER", "critical", key, "No header row detected.")
                return
            fieldnames = [str(c).strip() for c in reader.fieldnames]
            if len(set(fieldnames)) != len(fieldnames):
                self.warn("DUPLICATE_COLUMNS", "high", key, "Duplicate column names detected.")
            stats = {c: ColumnStats(c) for c in fieldnames}
            row_count = 0
            malformed = 0
            full_seen: set[bytes] = set()
            semantic_seen: set[bytes] = set()
            full_dups = 0
            semantic_dups = 0
            id_cols = [c for c in fieldnames if c == "_id" or ID_RE.search(c)]
            # Duplicate-event semantics retain grouping keys such as user/session.
            # Only record-level identifiers are excluded; otherwise identical events
            # in different sessions would be incorrectly counted as duplicates.
            record_id_cols = [c for c in fieldnames if c.lower() in {"_id", "record_id", "event_id", "row_id", "uuid", "guid"}]
            semantic_cols = [c for c in fieldnames if c not in record_id_cols]
            user_cols = [c for c in fieldnames if USER_RE.search(c)]
            session_cols = [c for c in fieldnames if SESSION_RE.search(c)]
            event_cols = [c for c in fieldnames if EVENT_RE.search(c)]
            time_cols = [c for c in fieldnames if TIME_RE.search(c)]
            label_cols = [c for c in fieldnames if LABEL_RE.search(c)]
            user_col = user_cols[0] if user_cols else None
            session_col = session_cols[0] if session_cols else None
            event_col = event_cols[0] if event_cols else None
            timestamp_col = time_cols[0] if time_cols else None
            event_counts = Counter()
            modality_counts = Counter()
            user_rows = Counter()
            user_sessions: dict[str, set[str]] = defaultdict(set)
            session_rows = Counter()
            session_users: dict[str, set[str]] = defaultdict(set)
            session_modalities: dict[str, set[str]] = defaultdict(set)
            session_first_ts: dict[str, datetime] = {}
            session_last_ts: dict[str, datetime] = {}
            session_previous_ts: dict[str, datetime] = {}
            backward_events = 0
            backward_sessions: set[str] = set()
            duplicate_session_timestamps = 0
            seen_session_timestamps: set[bytes] = set()
            sample_written = 0

            for row in reader:
                row_count += 1
                if None in row or any(c not in row for c in fieldnames):
                    malformed += 1
                normalized = {c: (row.get(c) or "") for c in fieldnames}
                for c in fieldnames:
                    stats[c].update(normalized[c], self.unique_cap)
                packed = "\x1f".join(normalized[c] for c in fieldnames).encode("utf-8", "replace")
                h = hashlib.blake2b(packed, digest_size=16).digest()
                if h in full_seen:
                    full_dups += 1
                else:
                    full_seen.add(h)
                sem_packed = "\x1f".join(normalized[c] for c in semantic_cols).encode("utf-8", "replace")
                sh = hashlib.blake2b(sem_packed, digest_size=16).digest()
                if sh in semantic_seen:
                    semantic_dups += 1
                else:
                    semantic_seen.add(sh)

                user = normalized.get(user_col, "") if user_col else ""
                session = normalized.get(session_col, "") if session_col else ""
                event = normalized.get(event_col, "") if event_col else ""
                if user:
                    user_rows[user] += 1
                if session:
                    session_rows[session] += 1
                if user and session:
                    user_sessions[user].add(session)
                    session_users[session].add(user)
                if event:
                    event_counts[event] += 1
                    el = event.lower()
                    if el in {"keydown", "keyup", "keypress"} or "key" in el:
                        modality = "keyboard"
                    elif "mouse" in el or el in {"mousedown", "mouseup", "mousemove", "mouseover", "mouseout", "click", "dblclick", "wheel", "mousewheel"}:
                        modality = "mouse"
                    else:
                        modality = "other"
                    modality_counts[modality] += 1
                    if session:
                        session_modalities[session].add(modality)

                if timestamp_col:
                    dt = parse_datetime(normalized.get(timestamp_col, ""))
                    if dt is not None and session:
                        if session not in session_first_ts or dt < session_first_ts[session]:
                            session_first_ts[session] = dt
                        if session not in session_last_ts or dt > session_last_ts[session]:
                            session_last_ts[session] = dt
                        prev = session_previous_ts.get(session)
                        if prev is not None and dt < prev:
                            backward_events += 1
                            backward_sessions.add(session)
                        session_previous_ts[session] = dt
                        sig = hashlib.blake2b(f"{session}\x1f{dt.isoformat()}".encode(), digest_size=16).digest()
                        if sig in seen_session_timestamps:
                            duplicate_session_timestamps += 1
                        else:
                            seen_session_timestamps.add(sig)

                if sample_written < self.sample_rows:
                    out = {"file": key, "sample_index": sample_written + 1}
                    out.update({c: sanitize(normalized[c], c, self.salt) for c in fieldnames})
                    self.samples.append(out)
                    sample_written += 1

            schema_cols = []
            for c in fieldnames:
                s = stats[c]
                distinct = None if s.unique_capped else len(s.unique_values)
                schema_cols.append({
                    "column": c,
                    "inferred_type": s.inferred_type(),
                    "missing": s.missing,
                    "nonmissing": s.nonmissing,
                    "distinct": distinct if distinct is not None else f">{self.unique_cap}",
                    "top_values": s.top_values.most_common(10),
                    "min_number": s.min_number,
                    "max_number": s.max_number,
                    "min_datetime": s.min_datetime.isoformat(sep=" ") if s.min_datetime else None,
                    "max_datetime": s.max_datetime.isoformat(sep=" ") if s.max_datetime else None,
                })
                self.missing_rows.append({
                    "file": key,
                    "column": c,
                    "missing_count": s.missing,
                    "missing_percent": round(100 * s.missing / max(1, row_count), 6),
                })
                uniqueness = (distinct / row_count) if distinct is not None and row_count else None
                candidate_reason = []
                risk = "low"
                if c == "_id" or ID_RE.search(c):
                    candidate_reason.append("identifier-like column name")
                    risk = "critical" if c in {"_id", "user", "user_id", "subject"} else "high"
                if uniqueness is not None and uniqueness >= 0.98:
                    candidate_reason.append("near-unique values")
                    risk = "high"
                if c.lower() in {"ua", "user_agent", "screen", "screenh", "windoww", "windowh", "documentw", "documenth"}:
                    candidate_reason.append("device/browser fingerprint proxy")
                    risk = "high"
                if candidate_reason:
                    self.identifier_rows.append({
                        "file": key,
                        "column": c,
                        "distinct": distinct if distinct is not None else f">{self.unique_cap}",
                        "row_count": row_count,
                        "uniqueness_ratio": round(uniqueness, 6) if uniqueness is not None else "capped",
                        "risk": risk,
                        "reason": "; ".join(candidate_reason),
                        "recommended_action": "retain only for grouping/audit; exclude from model features",
                    })

            self.schemas[key] = {
                "encoding": encoding,
                "delimiter": "TAB" if delimiter == "\t" else delimiter,
                "row_count": row_count,
                "column_count": len(fieldnames),
                "columns": schema_cols,
            }
            self.duplicate_rows.append({
                "file": key,
                "row_count": row_count,
                "exact_full_row_duplicates": full_dups,
                "semantic_duplicates_excluding_identifier_columns": semantic_dups,
                "duplicate_session_timestamp_pairs": duplicate_session_timestamps,
                "malformed_rows": malformed,
            })
            for c in time_cols:
                s = stats[c]
                self.timestamp_rows.append({
                    "file": key,
                    "column": c,
                    "parsed_count": s.datetime_like,
                    "parse_failure_count": s.nonmissing - s.datetime_like,
                    "min_timestamp": s.min_datetime.isoformat(sep=" ") if s.min_datetime else "",
                    "max_timestamp": s.max_datetime.isoformat(sep=" ") if s.max_datetime else "",
                    "backward_events_within_session": backward_events if c == timestamp_col else "",
                    "sessions_with_backward_events": len(backward_sessions) if c == timestamp_col else "",
                })
            for c in label_cols:
                s = stats[c]
                self.label_rows.append({
                    "file": key,
                    "column": c,
                    "role": "explicit label candidate",
                    "distinct": len(s.unique_values) if not s.unique_capped else f">{self.unique_cap}",
                    "top_values": json.dumps(s.top_values.most_common(20), ensure_ascii=False),
                })
            if user_col:
                self.label_rows.append({
                    "file": key,
                    "column": user_col,
                    "role": "identity label / grouping key; never use as predictor",
                    "distinct": len(stats[user_col].unique_values) if not stats[user_col].unique_capped else f">{self.unique_cap}",
                    "top_values": json.dumps(stats[user_col].top_values.most_common(20), ensure_ascii=False),
                })
            for user, rows in sorted(user_rows.items()):
                sess = user_sessions.get(user, set())
                self.user_session_rows.append({
                    "file": key,
                    "user": sanitize(user, user_col or "user", self.salt),
                    "original_user_value": user,
                    "row_count": rows,
                    "session_count": len(sess),
                    "min_rows_per_session": min((session_rows[s] for s in sess), default=0),
                    "median_rows_per_session": self._median([session_rows[s] for s in sess]),
                    "max_rows_per_session": max((session_rows[s] for s in sess), default=0),
                })

            sessions_both = sum(1 for m in session_modalities.values() if "keyboard" in m and "mouse" in m)
            sessions_keyboard_only = sum(1 for m in session_modalities.values() if m == {"keyboard"})
            sessions_mouse_only = sum(1 for m in session_modalities.values() if m == {"mouse"})
            self.file_summaries[key] = {
                "row_count": row_count,
                "columns": fieldnames,
                "user_col": user_col,
                "session_col": session_col,
                "event_col": event_col,
                "timestamp_col": timestamp_col,
                "users": set(user_rows),
                "sessions": set(session_rows),
                "session_users": session_users,
                "session_first_ts": session_first_ts,
                "session_last_ts": session_last_ts,
                "event_counts": event_counts,
                "modality_counts": modality_counts,
                "sessions_both": sessions_both,
                "sessions_keyboard_only": sessions_keyboard_only,
                "sessions_mouse_only": sessions_mouse_only,
                "constant_columns": [c for c, s in stats.items() if not s.unique_capped and len(s.unique_values) <= 1],
            }

            if malformed:
                self.warn("MALFORMED_ROWS", "high", key, f"Detected {malformed} malformed rows.")
            if backward_sessions:
                self.warn("NON_MONOTONIC_TIMESTAMPS", "high", key, f"{len(backward_sessions)} sessions contain backward timestamp steps; sort within session before feature extraction.")
            if duplicate_session_timestamps:
                self.warn("DUPLICATE_TIMESTAMPS", "medium", key, f"{duplicate_session_timestamps} duplicate session-timestamp pairs detected.")
            if semantic_dups:
                self.warn("SEMANTIC_DUPLICATES", "medium", key, f"{semantic_dups} duplicate rows after excluding identifier-like columns.")
            if "cursor" in stats and not stats["cursor"].unique_capped and len(stats["cursor"].unique_values) <= 1:
                self.warn("CONSTANT_CURSOR", "medium", key, "cursor is constant and should not be used as a feature.")
            if any(u.lower() == "alluserspreauth" for u in user_rows):
                self.warn("SPECIAL_PREAUTH_LABEL", "high", key, "Alluserspreauth is a shared pre-authentication context label, not a normal enrolled identity.")
            if user_col and any(len(v) > 1 for v in session_users.values()):
                self.warn("SESSION_USER_CONFLICT", "critical", key, "At least one session maps to multiple user values.")
        finally:
            fh.close()

    @staticmethod
    def _median(values: list[int]) -> float:
        if not values:
            return 0.0
        x = sorted(values)
        n = len(x)
        return float(x[n // 2]) if n % 2 else (x[n // 2 - 1] + x[n // 2]) / 2

    def cross_file_checks(self) -> None:
        session_files = [(k, v) for k, v in self.file_summaries.items() if v.get("session_col")]
        checks = []
        for i in range(len(session_files)):
            for j in range(i + 1, len(session_files)):
                ka, a = session_files[i]
                kb, b = session_files[j]
                sa, sb = a["sessions"], b["sessions"]
                inter = sa & sb
                mismatch = 0
                for s in inter:
                    ua = a["session_users"].get(s, set())
                    ub = b["session_users"].get(s, set())
                    if ua and ub and ua != ub:
                        mismatch += 1
                checks.append({
                    "file_a": ka,
                    "file_b": kb,
                    "sessions_a": len(sa),
                    "sessions_b": len(sb),
                    "intersection": len(inter),
                    "a_only": len(sa - sb),
                    "b_only": len(sb - sa),
                    "user_mapping_mismatches": mismatch,
                })
                if sa - sb or sb - sa:
                    self.warn("SESSION_SET_MISMATCH", "high", f"{ka} vs {kb}", f"Session sets differ: {len(sa-sb)} only in A and {len(sb-sa)} only in B.")
                if mismatch:
                    self.warn("CROSS_FILE_USER_MISMATCH", "critical", f"{ka} vs {kb}", f"{mismatch} shared sessions map to different users.")
        self.cross_file["session_join_checks"] = checks

        # Compare first timestamps for shared sessions. This detects systematic
        # timezone/clock offsets without attempting a dangerous nearest-time join.
        timestamp_checks = []
        for i in range(len(session_files)):
            for j in range(i + 1, len(session_files)):
                ka, a = session_files[i]
                kb, b = session_files[j]
                common = set(a.get("session_first_ts", {})) & set(b.get("session_first_ts", {}))
                offsets = []
                for s in common:
                    offsets.append((a["session_first_ts"][s] - b["session_first_ts"][s]).total_seconds())
                if offsets:
                    vals = sorted(offsets)
                    def pct(q: float) -> float:
                        idx = min(len(vals)-1, max(0, round(q*(len(vals)-1))))
                        return vals[idx]
                    check = {
                        "file_a": ka,
                        "file_b": kb,
                        "shared_sessions": len(vals),
                        "offset_definition": "first_timestamp_A_minus_first_timestamp_B_seconds",
                        "min_seconds": min(vals),
                        "p05_seconds": pct(0.05),
                        "median_seconds": pct(0.50),
                        "p95_seconds": pct(0.95),
                        "max_seconds": max(vals),
                    }
                    timestamp_checks.append(check)
                    if abs(check["median_seconds"]) > 60:
                        self.warn("SYSTEMATIC_TIMESTAMP_OFFSET", "high", f"{ka} vs {kb}",
                                  f"Median first-timestamp offset is {check['median_seconds']:.3f} seconds; do not align by raw time without normalization.")
        self.cross_file["timestamp_alignment_checks"] = timestamp_checks
        self.cross_file["timestamp_alignment_note"] = (
            "Use session_id for validated joins. Treat systematic offsets as clock/timezone metadata, "
            "not as evidence that sessions are different."
        )

    def write_outputs(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv("inventory.csv", self.inventory)
        (self.output_dir / "file_schemas.json").write_text(json.dumps(self.schemas, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        self._write_csv("safe_sample_rows.csv", self.samples)
        self._write_csv("user_session_statistics.csv", self.user_session_rows)
        self._write_csv("missing_values.csv", self.missing_rows)
        self._write_csv("duplicate_analysis.csv", self.duplicate_rows)
        self._write_csv("timestamp_analysis.csv", self.timestamp_rows)
        self._write_csv("label_analysis.csv", self.label_rows)
        self._write_csv("candidate_identifiers.csv", self.identifier_rows)
        self._write_csv("warnings_and_failures.csv", self.warnings)
        (self.output_dir / "cross_file_integrity.json").write_text(json.dumps(self.cross_file, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (self.output_dir / "leakage_risk_report.md").write_text(self.leakage_report(), encoding="utf-8")
        (self.output_dir / "recommended_splitting_strategy.md").write_text(self.split_report(), encoding="utf-8")
        (self.output_dir / "recommended_windowing_alternatives.md").write_text(self.window_report(), encoding="utf-8")
        (self.output_dir / "stage1_findings_ar.md").write_text(self.arabic_findings_report(), encoding="utf-8")
        (self.output_dir / "stage1_summary.json").write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (self.output_dir / "execution_log.txt").write_text("\n".join(self.logs) + "\n", encoding="utf-8")
        manifest = []
        for f in sorted(self.output_dir.iterdir()):
            if f.is_file() and f.name != "manifest_sha256.csv":
                manifest.append({"file": f.name, "bytes": f.stat().st_size, "sha256": sha256_file(f)})
        self._write_csv("manifest_sha256.csv", manifest)
        zip_path = self.output_dir.parent / f"{self.output_dir.name}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for f in sorted(self.output_dir.iterdir()):
                if f.is_file():
                    z.write(f, arcname=f.name)
        self.log(f"Created review ZIP: {zip_path}")
        return zip_path

    def _write_csv(self, name: str, rows: list[dict]) -> None:
        path = self.output_dir / name
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = []
        for row in rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    def summary(self) -> dict:
        files = {}
        for k, v in self.file_summaries.items():
            files[k] = {
                "rows": v["row_count"],
                "columns": v["columns"],
                "unique_user_values": len(v["users"]),
                "unique_sessions": len(v["sessions"]),
                "event_counts": dict(v["event_counts"]),
                "modality_counts": dict(v["modality_counts"]),
                "sessions_both_modalities": v["sessions_both"],
                "sessions_keyboard_only": v["sessions_keyboard_only"],
                "sessions_mouse_only": v["sessions_mouse_only"],
                "constant_columns": v["constant_columns"],
            }
        return {
            "pilot_version": VERSION,
            "input": str(self.input_path),
            "input_sha256": sha256_file(self.input_path) if self.input_path.is_file() else None,
            "read_only": True,
            "training_performed": False,
            "files": files,
            "warning_count": len(self.warnings),
            "warnings_by_severity": dict(Counter(w["severity"] for w in self.warnings)),
            "decision": "Proceed only after human review of Stage 1 outputs.",
        }


    def arabic_findings_report(self) -> str:
        event_files = [(k, v) for k, v in self.file_summaries.items() if v.get("event_col")]
        meta_files = [(k, v) for k, v in self.file_summaries.items() if not v.get("event_col") and v.get("session_col")]
        lines = [
            "# نتائج Keystroke–Mouse Stage 1",
            "",
            "## قرار سلامة المصدر",
            "",
            "تم فحص المصدر بوضع القراءة فقط، ولم تُعدَّل الملفات الأصلية ولم يُنفَّذ أي تدريب أو استخراج ميزات.",
            "",
        ]
        if meta_files:
            k, v = meta_files[0]
            lines += [f"- ملف المعلومات: `{k}` — {v['row_count']:,} سجلاً و{len(v['sessions']):,} جلسة وصفية."]
        if event_files:
            k, v = event_files[0]
            lines += [
                f"- ملف الأحداث: `{k}` — {v['row_count']:,} حدثاً ضمن {len(v['sessions']):,} جلسة ذات أحداث.",
                f"- أحداث لوحة المفاتيح: {v['modality_counts'].get('keyboard', 0):,}.",
                f"- أحداث الفأرة: {v['modality_counts'].get('mouse', 0):,}.",
                f"- الجلسات متعددة الوسائط: {v['sessions_both']:,}؛ لوحة مفاتيح فقط: {v['sessions_keyboard_only']:,}؛ فأرة فقط: {v['sessions_mouse_only']:,}.",
                f"- عدد قيم المستخدم الظاهرة: {len(v['users'])}، وتشمل الوسم الخاص `Alluserspreauth`؛ لذلك عدد المستخدمين المسجلين الفعليين هو 11.",
            ]
        checks = self.cross_file.get("session_join_checks", [])
        if checks:
            c = checks[0]
            lines += [
                "",
                "## سلامة الربط بين الملفين",
                "",
                f"- الجلسات المشتركة: {c['intersection']:,}.",
                f"- جلسات موجودة في ملف المعلومات بلا أحداث: {c['a_only']:,}.",
                f"- جلسات أحداث بلا سجل معلومات: {c['b_only']:,}.",
                f"- تعارضات ربط المستخدم بالجلسة: {c['user_mapping_mismatches']:,}.",
            ]
        tchecks = self.cross_file.get("timestamp_alignment_checks", [])
        if tchecks:
            c = tchecks[0]
            lines += [
                "",
                "## الطوابع الزمنية",
                "",
                f"يوجد فرق زمني منهجي بين وقت المعلومات وأول حدث، ووسيط الفرق {c['median_seconds']:.3f} ثانية تقريباً (نحو ساعة). يجب الربط بواسطة `session_id` وعدم مطابقة السجلات بأقرب وقت.",
            ]
        lines += [
            "",
            "## أهم مخاطر التسرب",
            "",
            "- استبعاد `_id` و`user` و`session_id` وحقول المتصفح والشاشة من مدخلات النموذج.",
            "- تقسيم الجلسات قبل بناء النوافذ، ومنع انتقال نوافذ الجلسة نفسها بين Train وValidation وTest.",
            "- فرز الأحداث داخل كل جلسة زمنياً قبل حساب dwell/flight والسرعة والتسارع.",
            "- عدم اعتبار `Alluserspreauth` مستخدماً مسجلاً أو فئة impostor عادية قبل توثيق معناه.",
            "- إنشاء المطالبات Genuine/Impostor بعد تجميد تقسيم الجلسات، وحفظ تنبؤات Validation وTest على مستوى claim.",
            "",
            "## القرار",
            "",
            "المجموعة صالحة للانتقال إلى تصميم Stage 2 بعد مراجعة بشرية لهذه المخرجات، مع عدم بدء التدريب بعد. التمثيل الأنسب مبدئياً هو نوافذ زمنية متزامنة متعددة الوسائط مع أقنعة للوسائط المفقودة، إلى جانب مسارين منفصلين للوحة المفاتيح والفأرة للمقارنة والدمج المتأخر.",
        ]
        return "\n".join(lines) + "\n"

    def leakage_report(self) -> str:
        return """# Leakage-Risk Report

## Critical controls

1. Split by `session_id` before any window generation. All windows from one session must remain in one partition.
2. Keep the same enrolled users across Train/Validation/Test for the primary verification protocol, but use disjoint sessions and chronological ordering within each user.
3. Never use `_id`, `user`, `session_id`, raw timestamps, user-agent strings, screen/window/document dimensions, file path, or row order as model predictors.
4. Fit scaling, imputation, encoding, feature selection, calibration, and threshold selection using Training/Validation only. Test must remain untouched.
5. Generate genuine and impostor claims after the partition is frozen. A source window must not appear in both genuine and impostor pools across partitions.
6. Deduplicate before splitting using a semantic event signature that excludes record identifiers. Duplicate groups must be assigned atomically to one partition.
7. Do not randomly split events or overlapping windows. Overlap creates near-duplicate samples and inflates performance.
8. Treat `Alluserspreauth` as a special shared context, not as an enrolled user, until its semantics and linkage are validated.

## Dataset-specific risks detected

- The metadata and tracking files may contain different session sets; use an inner join only for event-bearing sessions and retain a separate audit of metadata-only sessions.
- Some timestamp sequences may be non-monotonic. Sort by `(session_id, timestamp, stable_row_index)` before pairing keys or computing mouse derivatives.
- Device/browser fields can act as proxy identifiers and create shortcut learning.
- Keyboard key codes can reveal text/task content. Evaluate both timing-only and timing-plus-key-category variants; do not allow exact text identity to dominate user identity.
- Variable session sizes can cause large sessions to dominate. Use per-user/per-session weighting and clustered confidence intervals.
"""

    def split_report(self) -> str:
        return """# Recommended Splitting Strategy

## Primary protocol: known-user continuous authentication

- Exclude `Alluserspreauth` from the enrolled-user identity set in the primary experiment.
- For each authenticated user, order event-bearing sessions chronologically.
- Allocate sessions approximately 60% Training, 20% Validation, 20% Test, with at least one session in each split.
- Small-session rule:
  - 3 sessions → 1/1/1.
  - 4 sessions → 2/1/1.
  - 5 sessions → 3/1/1.
  - 6+ sessions → chronological 60/20/20 with deterministic rounding.
- Freeze the session manifest before creating windows or claims.
- Build user-specific genuine claims from that user's windows in the current partition.
- Build impostor claims only from other users' windows in the same partition.
- Validation is used for calibration and per-user threshold selection; Test is used once for final reporting.

## Secondary robustness protocol

A user-disjoint/open-set split can be reported separately as a domain-generalization experiment, but it must not replace the primary known-user verification protocol because unseen users cannot have standard enrolled templates without an enrollment phase.

## Claim protocols

1. Balanced Genuine–Impostor 1:1: deterministic stratified sampling within each claimed user and split.
2. Realistic All-Impostor Claims: score every eligible other-user window against each claimed user, while preserving source-window identifiers.

Persist `user_id`, `session_id`, `window_id`, `claimed_user_id`, `actual_user_id`, `is_genuine`, raw score, calibrated score, threshold, and prediction for Validation and Test.
"""

    def window_report(self) -> str:
        return """# Recommended Windowing Alternatives

No single representation should be fixed before Stage 1 review. Evaluate the following alternatives without changing the Train/Validation/Test session manifest.

## A. Keyboard event-pair windows

- Reconstruct keydown/keyup pairs after stable sorting.
- Candidate sizes: 40, 80, and 120 valid key events, with non-overlapping windows for the primary benchmark.
- Features: dwell, press–press, release–press, flight, typing speed, pauses, burst lengths, backspace/correction behaviour, and robust distribution summaries.

## B. Mouse trajectory windows

- Candidate time windows: 5, 10, and 20 seconds; or 50, 100, and 200 movement events.
- Segment around movement/click/drag episodes.
- Features: velocity, acceleration, jerk, curvature, direction changes, path efficiency, click duration, inter-click interval, idle ratio, and drag behaviour.

## C. Synchronized multimodal time windows

- Candidate durations: 10, 20, and 30 seconds.
- Create one keyboard branch and one mouse branch using the same temporal boundaries.
- Permit missing-modality masks rather than discarding all single-modality windows.
- Compare early fusion, late score fusion, and a hybrid two-branch model.

## D. Session-level aggregation

Use only as a classical baseline because session lengths are highly variable and the number of sessions per user can be small. Do not treat individual events as independent samples.

## Fairness controls

- Primary results should use non-overlapping windows.
- Any overlapping-window sensitivity test must remain secondary and report the overlap ratio.
- Apply identical session splits to all models and fusion methods.
"""


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only Stage 1 audit for keystroke/mouse datasets")
    p.add_argument("--input", required=True, type=Path, help="Input ZIP file or directory")
    p.add_argument("--output", type=Path, default=Path("stage1_review"), help="Output directory")
    p.add_argument("--sample-rows", type=int, default=5, help="Safe sample rows per file")
    p.add_argument("--unique-cap", type=int, default=200000, help="Maximum unique values retained per column")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    audit = Audit(args.input.resolve(), args.output.resolve(), max(1, args.sample_rows), max(1000, args.unique_cap))
    audit.log(f"Starting Stage 1 pilot v{VERSION}")
    audit.log(f"Input: {audit.input_path}")
    audit.log("Safety mode: read-only; no training; no source extraction or modification.")
    try:
        items = audit.discover()
        if not items:
            audit.warn("NO_SUPPORTED_FILES", "critical", str(audit.input_path), "No CSV/TSV/TXT files found.")
        for item in items:
            try:
                audit.audit_file(item)
            except Exception as exc:
                audit.warn("FILE_AUDIT_FAILURE", "critical", item.member, f"{type(exc).__name__}: {exc}")
                audit.logs.append(traceback.format_exc())
        audit.cross_file_checks()
        zip_path = audit.write_outputs()
        print(json.dumps({"status": "ok", "review_zip": str(zip_path), "output_dir": str(audit.output_dir)}, ensure_ascii=False))
        return 0
    except Exception as exc:
        audit.warn("FATAL", "critical", str(audit.input_path), f"{type(exc).__name__}: {exc}")
        audit.logs.append(traceback.format_exc())
        audit.output_dir.mkdir(parents=True, exist_ok=True)
        (audit.output_dir / "execution_log.txt").write_text("\n".join(audit.logs), encoding="utf-8")
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

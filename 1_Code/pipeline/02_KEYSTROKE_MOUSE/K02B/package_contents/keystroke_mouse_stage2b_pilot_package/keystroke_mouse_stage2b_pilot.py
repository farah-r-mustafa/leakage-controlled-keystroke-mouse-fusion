#!/usr/bin/env python3
"""
Keystroke–Mouse Stage 2B — Window Materialization and Leakage-Safe Feature Engineering

Design goals
------------
* Read-only access to the original dataset.
* Reuse the frozen Stage 2A session split and 60-second segment manifest.
* Materialize three prespecified representations only:
    1) keyboard fixed windows: 5 valid key pairs, step 2 pairs;
    2) mouse fixed windows: 10 valid mouse strokes, step 5 strokes;
    3) synchronized multimodal windows: 10 s, step 5 s, minK=3, minM=3.
* Never normalize, select features, train models, create claims, calibrate scores,
  or inspect model performance.
* Keep identifiers as metadata and explicitly prohibit them from model inputs.
* Produce deterministic window IDs and lineage records.
* Use only the Python standard library.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Iterator, Sequence

VERSION = "2.0.0"
STAGE_NAME = "Keystroke–Mouse Stage 2B"

EXPECTED_FROZEN_HASH = "e35422f0e48f96b6d1e4da99c9063379a6a488dd2bfa3e99d3e6e68857a7613d"
EXPECTED_SEGMENT_HASH = "21498c57b490c0fdeef876f621f8c2b758c51942c527a5c534396de00595bec9"
EXPECTED_DUPLICATE_HASH = "3dde057c37deaf343bfd6863496449c50a2a80f6fdd638b25e74d02af292ef19"

KEY_EVENTS = {"keydown", "keyup"}
MOUSE_EVENTS = {"mousemove", "mouseout", "mouseover", "mousedown", "mouseup", "mousewheel"}
EVENT_HEADER = {"_id", "cursor", "timestamp", "xpos", "ypos", "key", "event", "user", "session_id"}
SPLITS = ("train", "validation", "test")

# Stage 2A naming is retained for reproducibility. For N=5 and step=2,
# the effective overlap is 3/5=60%; exact 50% is impossible for an odd count.
KEY_WINDOW_SIZE = 5
KEY_WINDOW_STEP = 2
MOUSE_WINDOW_SIZE = 10
MOUSE_WINDOW_STEP = 5
TIME_WINDOW_SECONDS = 10.0
TIME_WINDOW_STEP_SECONDS = 5.0
TIME_MIN_KEY = 3
TIME_MIN_MOUSE = 3
MAX_DWELL_SECONDS = 2.0

METADATA_COLUMNS = [
    "window_id", "window_family", "window_config", "split", "user_id",
    "session_id", "segment_id", "window_order", "window_start_time",
    "window_end_time", "window_start_epoch", "window_end_epoch",
    "source_start_index", "source_end_index", "keyboard_observed",
    "mouse_observed", "keyboard_eligible", "mouse_eligible",
    "eligible_any", "eligible_both", "primary_multimodal",
]


@dataclass
class CsvSource:
    name: str
    opener: Callable[[], BinaryIO]


class Stage2BError(RuntimeError):
    pass


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_dt(value: str) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        # Conservative fallback for common fractional-second formats.
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                pass
    return None


def fmt_epoch(epoch: float | None) -> str:
    if epoch is None or not math.isfinite(epoch):
        return ""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def clean_float(value: float | int | None, digits: int = 9):
    if value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(x):
        return ""
    if abs(x) < 0.5 * 10 ** (-digits):
        x = 0.0
    return round(x, digits)


def percentile(values: Sequence[float], p: float) -> float | None:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def stats(values: Sequence[float], prefix: str, scale: float = 1.0) -> dict[str, object]:
    vals = [float(v) * scale for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {
            f"{prefix}_mean": "", f"{prefix}_std": "", f"{prefix}_median": "",
            f"{prefix}_p10": "", f"{prefix}_p90": "", f"{prefix}_p95": "",
            f"{prefix}_min": "", f"{prefix}_max": "", f"{prefix}_iqr": "",
            f"{prefix}_cv": "",
        }
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    q25 = percentile(vals, 0.25)
    q75 = percentile(vals, 0.75)
    return {
        f"{prefix}_mean": clean_float(mean),
        f"{prefix}_std": clean_float(std),
        f"{prefix}_median": clean_float(percentile(vals, 0.50)),
        f"{prefix}_p10": clean_float(percentile(vals, 0.10)),
        f"{prefix}_p90": clean_float(percentile(vals, 0.90)),
        f"{prefix}_p95": clean_float(percentile(vals, 0.95)),
        f"{prefix}_min": clean_float(min(vals)),
        f"{prefix}_max": clean_float(max(vals)),
        f"{prefix}_iqr": clean_float((q75 or 0.0) - (q25 or 0.0)),
        f"{prefix}_cv": clean_float(std / abs(mean)) if abs(mean) > 1e-12 else "",
    }


def angle_difference(a: float, b: float) -> float:
    d = b - a
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\r\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def stable_id(parts: Sequence[object], prefix: str) -> str:
    text = "|".join(str(x) for x in parts)
    return f"{prefix}_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}"


def user_sort_key(user: str) -> tuple[int, str]:
    if user.lower().startswith("user"):
        try:
            return (int(user[4:]), user)
        except ValueError:
            pass
    return (10**9, user)


def key_category(raw: str) -> str:
    """Map JavaScript-style numeric key codes to coarse, non-content categories."""
    try:
        code = int(float((raw or "").strip()))
    except ValueError:
        return "other"
    if code in {8, 46}:
        return "correction"
    if code in {13, 9}:
        return "enter_tab"
    if code in {16, 17, 18, 20, 91, 92, 93}:
        return "modifier"
    if code in {32}:
        return "space"
    if code in {33, 34, 35, 36, 37, 38, 39, 40, 45}:
        return "navigation"
    if 112 <= code <= 123:
        return "function"
    if 48 <= code <= 90 or 96 <= code <= 111:
        return "alphanumeric"
    return "other"


def discover_event_source(input_path: Path) -> CsvSource:
    def header_of_binary(opener: Callable[[], BinaryIO]) -> set[str]:
        raw = opener()
        try:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
            line = text.readline()
        finally:
            try:
                raw.close()
            except Exception:
                pass
        delimiter = "\t" if line.count("\t") >= line.count(",") else ","
        return {x.strip() for x in next(csv.reader([line], delimiter=delimiter), [])}

    if input_path.is_file() and zipfile.is_zipfile(input_path):
        zf = zipfile.ZipFile(input_path, "r")
        candidates: list[tuple[str, set[str]]] = []
        for name in zf.namelist():
            if name.lower().endswith((".csv", ".tsv", ".txt")) and not name.endswith("/"):
                def make_opener(n=name):
                    return zf.open(n, "r")
                try:
                    header = header_of_binary(make_opener)
                except Exception:
                    continue
                candidates.append((name, header))
        for name, header in candidates:
            if EVENT_HEADER.issubset(header):
                return CsvSource(name=name, opener=lambda n=name, archive=zf: archive.open(n, "r"))
        zf.close()
        raise Stage2BError("Could not locate EVTRACKTRACK-like event table inside the ZIP file.")

    files = [input_path] if input_path.is_file() else list(input_path.rglob("*"))
    for p in files:
        if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".txt"}:
            try:
                header = header_of_binary(lambda q=p: q.open("rb"))
            except Exception:
                continue
            if EVENT_HEADER.issubset(header):
                return CsvSource(name=str(p), opener=lambda q=p: q.open("rb"))
    raise Stage2BError("Could not locate an event table with the required columns.")


def read_csv_dicts(source: CsvSource) -> Iterator[dict[str, str]]:
    raw = source.opener()
    text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
    first = text.readline()
    delimiter = "\t" if first.count("\t") >= first.count(",") else ","
    fieldnames = [x.strip() for x in next(csv.reader([first], delimiter=delimiter), [])]
    reader = csv.DictReader(text, fieldnames=fieldnames, delimiter=delimiter)
    try:
        for row in reader:
            yield {str(k).strip(): (v if v is not None else "") for k, v in row.items()}
    finally:
        text.close()


def load_csv_from_dir_or_zip(base: Path, filename: str) -> tuple[list[dict], bytes]:
    if base.is_file() and zipfile.is_zipfile(base):
        with zipfile.ZipFile(base, "r") as zf:
            matches = [n for n in zf.namelist() if Path(n).name.lower() == filename.lower()]
            if not matches:
                raise Stage2BError(f"Missing {filename} in Stage 2A review ZIP.")
            data = zf.read(matches[0])
    else:
        p = base / filename
        if not p.exists():
            matches = list(base.rglob(filename)) if base.exists() else []
            if not matches:
                raise Stage2BError(f"Missing frozen input: {filename}")
            p = matches[0]
        data = p.read_bytes()
    text = data.decode("utf-8-sig", errors="strict")
    rows = list(csv.DictReader(io.StringIO(text)))
    return rows, data


class Pilot:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.input_path = Path(args.input).expanduser().resolve()
        self.output_dir = Path(args.output).expanduser().resolve()
        self.script_dir = Path(__file__).resolve().parent
        self.frozen_base = Path(args.frozen_inputs).expanduser().resolve() if args.frozen_inputs else self.script_dir / "stage2a_frozen_inputs"
        self.warnings: list[dict] = []
        self.failures: list[dict] = []
        self.log_lines: list[str] = []
        self.conn: sqlite3.Connection | None = None
        self.sqlite_path: Path | None = None
        self.core_manifest: dict[str, dict] = {}
        self.segments: list[dict] = []
        self.segment_events_cache: dict[str, list[dict]] = {}
        self.feature_rows: dict[str, dict[str, list[dict]]] = {
            "keyboard": {s: [] for s in SPLITS},
            "mouse": {s: [] for s in SPLITS},
            "multimodal": {s: [] for s in SPLITS},
        }
        self.window_manifest: list[dict] = []
        self.sequence_records: dict[str, dict[str, list[dict]]] = {
            "keyboard": {s: [] for s in SPLITS},
            "mouse": {s: [] for s in SPLITS},
            "multimodal": {s: [] for s in SPLITS},
        }
        self.start_clock = time.time()

    def log(self, message: str) -> None:
        line = f"[{now_text()}] {message}"
        self.log_lines.append(line)
        print(line, flush=True)

    def warn(self, code: str, severity: str, message: str, context: str = "") -> None:
        self.warnings.append({"type": "warning", "code": code, "severity": severity, "message": message, "context": context})
        self.log(f"WARNING {code}: {message}")

    def fail(self, code: str, message: str, context: str = "") -> None:
        self.failures.append({"type": "failure", "code": code, "severity": "critical", "message": message, "context": context})
        self.log(f"FAILURE {code}: {message}")

    def prepare_output(self) -> None:
        if self.output_dir.exists():
            if not self.args.overwrite:
                raise Stage2BError(f"Output directory already exists: {self.output_dir}. Use --overwrite to replace it.")
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"Starting {STAGE_NAME} v{VERSION}")
        self.log(f"Input: {self.input_path}")
        self.log(f"Frozen Stage 2A inputs: {self.frozen_base}")
        self.log(f"Output: {self.output_dir}")

    def load_and_verify_frozen_inputs(self) -> dict:
        manifest_rows, manifest_bytes = load_csv_from_dir_or_zip(self.frozen_base, "frozen_session_manifest.csv")
        segment_rows, segment_bytes = load_csv_from_dir_or_zip(self.frozen_base, "primary_segment_manifest_60s.csv")
        duplicate_rows, duplicate_bytes = load_csv_from_dir_or_zip(self.frozen_base, "semantic_duplicate_groups.csv")
        hashes = {
            "frozen_session_manifest.csv": sha256_bytes(manifest_bytes),
            "primary_segment_manifest_60s.csv": sha256_bytes(segment_bytes),
            "semantic_duplicate_groups.csv": sha256_bytes(duplicate_bytes),
        }
        expected = {
            "frozen_session_manifest.csv": EXPECTED_FROZEN_HASH,
            "primary_segment_manifest_60s.csv": EXPECTED_SEGMENT_HASH,
            "semantic_duplicate_groups.csv": EXPECTED_DUPLICATE_HASH,
        }
        for name, actual in hashes.items():
            if actual != expected[name]:
                raise Stage2BError(f"Frozen input hash mismatch for {name}: {actual} != {expected[name]}")

        core = [r for r in manifest_rows if r.get("primary_role") == "core_primary" and r.get("split") in SPLITS]
        self.core_manifest = {r["session_id"]: r for r in core}
        if len(self.core_manifest) != 81:
            raise Stage2BError(f"Expected 81 frozen core sessions, found {len(self.core_manifest)}")

        seen_segment_ids: set[str] = set()
        for r in segment_rows:
            sid = r.get("session_id", "")
            if sid not in self.core_manifest:
                continue
            if r.get("split") != self.core_manifest[sid]["split"]:
                raise Stage2BError(f"Segment split mismatch for session {sid}")
            seg_id = r.get("segment_id", "")
            if not seg_id or seg_id in seen_segment_ids:
                raise Stage2BError(f"Missing or duplicate segment_id: {seg_id}")
            seen_segment_ids.add(seg_id)
            start = parse_dt(r.get("start_time", ""))
            end = parse_dt(r.get("end_time", ""))
            if start is None or end is None or end < start:
                raise Stage2BError(f"Invalid segment boundary for {seg_id}")
            rr = dict(r)
            rr["start_epoch"] = start
            rr["end_epoch"] = end
            rr["segment_order_int"] = int(r.get("segment_order") or 0)
            self.segments.append(rr)
        self.segments.sort(key=lambda r: (user_sort_key(r["user"]), r["split"], r["session_id"], r["segment_order_int"]))
        if not self.segments:
            raise Stage2BError("No core segments were loaded from the frozen segment manifest.")

        # Copy binding inputs into the result for auditability.
        frozen_out = self.output_dir / "frozen_inputs"
        frozen_out.mkdir(parents=True, exist_ok=True)
        (frozen_out / "frozen_session_manifest.csv").write_bytes(manifest_bytes)
        (frozen_out / "primary_segment_manifest_60s.csv").write_bytes(segment_bytes)
        (frozen_out / "semantic_duplicate_groups.csv").write_bytes(duplicate_bytes)
        write_csv(self.output_dir / "frozen_input_hashes.csv", [
            {"file": k, "sha256": v, "expected_sha256": expected[k], "status": "PASS" if v == expected[k] else "FAIL"}
            for k, v in hashes.items()
        ], ["file", "sha256", "expected_sha256", "status"])
        self.log(f"Verified frozen split ({len(self.core_manifest)} core sessions) and {len(self.segments)} core segments")
        return {"hashes": hashes, "core_sessions": len(self.core_manifest), "core_segments": len(self.segments), "duplicate_groups": len(duplicate_rows)}

    def import_events(self) -> dict:
        source = discover_event_source(self.input_path)
        self.log(f"Detected event table: {source.name}")
        temp = tempfile.NamedTemporaryFile(prefix="stage2b_", suffix=".sqlite", delete=False)
        temp.close()
        self.sqlite_path = Path(temp.name)
        self.conn = sqlite3.connect(str(self.sqlite_path))
        c = self.conn.cursor()
        c.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE events_raw (
          source_row INTEGER PRIMARY KEY,
          raw_id TEXT,
          timestamp_text TEXT,
          ts_epoch REAL,
          xpos REAL,
          ypos REAL,
          key_code TEXT,
          event TEXT,
          user TEXT,
          session_id TEXT
        );
        """)
        core_ids = set(self.core_manifest)
        batch: list[tuple] = []
        total_source = retained = invalid_ts = invalid_xy = user_mismatch = 0
        for idx, row in enumerate(read_csv_dicts(source), start=1):
            total_source += 1
            sid = (row.get("session_id") or "").strip()
            if sid not in core_ids:
                continue
            ts = parse_dt(row.get("timestamp", ""))
            if ts is None:
                invalid_ts += 1
                continue
            try:
                x = float(row.get("xpos", ""))
                y = float(row.get("ypos", ""))
            except ValueError:
                x = y = float("nan")
                invalid_xy += 1
            user = (row.get("user") or "").strip()
            if user != self.core_manifest[sid]["user"]:
                user_mismatch += 1
            batch.append((
                idx, row.get("_id", ""), row.get("timestamp", ""), ts, x, y,
                row.get("key", ""), (row.get("event") or "").strip().lower(), user, sid,
            ))
            if len(batch) >= self.args.batch_size:
                c.executemany("INSERT INTO events_raw VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
                retained += len(batch)
                batch.clear()
        if batch:
            c.executemany("INSERT INTO events_raw VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
            retained += len(batch)
        self.conn.commit()
        c.executescript("""
        CREATE INDEX idx_events_session_time ON events_raw(session_id, ts_epoch, source_row);
        CREATE TABLE events_clean AS
          SELECT MIN(source_row) AS source_row, MIN(raw_id) AS raw_id,
                 timestamp_text, MIN(ts_epoch) AS ts_epoch, xpos, ypos,
                 key_code, event, user, session_id
          FROM events_raw
          GROUP BY timestamp_text, xpos, ypos, key_code, event, user, session_id;
        CREATE INDEX idx_clean_session_time ON events_clean(session_id, ts_epoch, source_row);
        """)
        self.conn.commit()
        raw_core = c.execute("SELECT COUNT(*) FROM events_raw").fetchone()[0]
        clean_core = c.execute("SELECT COUNT(*) FROM events_clean").fetchone()[0]
        removed = raw_core - clean_core
        if invalid_ts:
            self.fail("INVALID_TIMESTAMPS", f"{invalid_ts} core-session events had unparseable timestamps")
        if invalid_xy:
            self.warn("INVALID_COORDINATES", "high", f"{invalid_xy} core-session events had invalid coordinates")
        if user_mismatch:
            self.fail("USER_SESSION_CONFLICT", f"{user_mismatch} core-session rows conflicted with the frozen user mapping")
        self.log(f"Streamed {total_source} source rows; retained {raw_core} core rows; deduplicated to {clean_core}")
        return {
            "source_rows": total_source, "core_rows_raw": raw_core, "core_rows_clean": clean_core,
            "semantic_duplicates_removed_in_core": removed, "invalid_timestamps": invalid_ts,
            "invalid_coordinates": invalid_xy, "user_session_mismatches": user_mismatch,
        }

    def fetch_segment_events(self, seg: dict) -> list[dict]:
        assert self.conn is not None
        rows = self.conn.execute("""
          SELECT source_row, ts_epoch, timestamp_text, xpos, ypos, key_code, event, user, session_id
          FROM events_clean
          WHERE session_id=? AND ts_epoch>=? AND ts_epoch<=?
          ORDER BY ts_epoch, source_row
        """, (seg["session_id"], seg["start_epoch"], seg["end_epoch"])).fetchall()
        events = [
            {
                "source_row": int(r[0]), "ts": float(r[1]), "timestamp": r[2],
                "x": float(r[3]), "y": float(r[4]), "key": r[5], "event": r[6],
                "user": r[7], "session_id": r[8],
            }
            for r in rows
        ]
        expected = int(seg.get("event_count") or 0)
        if len(events) != expected:
            self.fail("SEGMENT_EVENT_COUNT_MISMATCH", f"{seg['segment_id']}: expected {expected}, recovered {len(events)}")
        return events

    def pair_keyboard(self, events: list[dict]) -> list[dict]:
        pending: dict[str, deque] = defaultdict(deque)
        pairs: list[dict] = []
        for e in events:
            if e["event"] == "keydown":
                pending[e["key"]].append(e)
            elif e["event"] == "keyup":
                if not pending[e["key"]]:
                    continue
                down = pending[e["key"]].popleft()
                dwell = e["ts"] - down["ts"]
                if 0 <= dwell <= MAX_DWELL_SECONDS:
                    pairs.append({
                        "down_ts": down["ts"], "up_ts": e["ts"], "dwell": dwell,
                        "key": down["key"], "category": key_category(down["key"]),
                        "down_source_row": down["source_row"], "up_source_row": e["source_row"],
                    })
        pairs.sort(key=lambda p: (p["down_ts"], p["down_source_row"], p["up_ts"]))
        return pairs

    def mouse_strokes(self, events: list[dict]) -> list[dict]:
        mouse = [e for e in events if e["event"] in MOUSE_EVENTS]
        strokes: list[dict] = []
        for a, b in zip(mouse, mouse[1:]):
            dt = b["ts"] - a["ts"]
            if dt <= 0:
                continue
            if not all(math.isfinite(v) for v in (a["x"], a["y"], b["x"], b["y"])):
                continue
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            dist = math.hypot(dx, dy)
            strokes.append({
                "start_ts": a["ts"], "end_ts": b["ts"], "dt": dt,
                "x0": a["x"], "y0": a["y"], "x1": b["x"], "y1": b["y"],
                "dx": dx, "dy": dy, "distance": dist, "velocity": dist / dt,
                "angle": math.atan2(dy, dx) if dist > 0 else None,
                "event_a": a["event"], "event_b": b["event"],
                "source_row_a": a["source_row"], "source_row_b": b["source_row"],
            })
        strokes.sort(key=lambda s: (s["end_ts"], s["source_row_b"]))
        return strokes

    def keyboard_features(self, pairs: Sequence[dict], window_start: float, window_end: float) -> dict:
        out: dict[str, object] = {}
        n = len(pairs)
        dwell = [p["dwell"] for p in pairs]
        down = [p["down_ts"] for p in pairs]
        press_press = [b - a for a, b in zip(down, down[1:])]
        release_press = [pairs[i]["down_ts"] - pairs[i - 1]["up_ts"] for i in range(1, n)]
        positive_pp = [x for x in press_press if x >= 0]
        categories = Counter(p["category"] for p in pairs)
        duration = max(window_end - window_start, 1e-9)

        out["k_pair_count"] = n
        out["k_window_span_s"] = clean_float(window_end - window_start)
        out["k_typing_rate_pairs_s"] = clean_float(n / duration)
        out["k_typing_rate_pairs_min"] = clean_float(n * 60.0 / duration)
        out.update(stats(dwell, "k_dwell_ms", 1000.0))
        out.update(stats(press_press, "k_press_press_ms", 1000.0))
        out.update(stats(release_press, "k_release_press_ms", 1000.0))
        out["k_overlap_ratio"] = clean_float(sum(x < 0 for x in release_press) / len(release_press)) if release_press else ""
        out["k_pause_500ms_ratio"] = clean_float(sum(x > 0.5 for x in positive_pp) / len(positive_pp)) if positive_pp else ""
        out["k_pause_1s_ratio"] = clean_float(sum(x > 1.0 for x in positive_pp) / len(positive_pp)) if positive_pp else ""
        out["k_pause_2s_ratio"] = clean_float(sum(x > 2.0 for x in positive_pp) / len(positive_pp)) if positive_pp else ""

        bursts: list[int] = []
        if n:
            current = 1
            for gap in press_press:
                if gap <= 1.0:
                    current += 1
                else:
                    bursts.append(current)
                    current = 1
            bursts.append(current)
        out["k_burst_count_1s"] = len(bursts)
        out["k_burst_mean_len_1s"] = clean_float(statistics.fmean(bursts)) if bursts else ""
        out["k_burst_max_len_1s"] = max(bursts) if bursts else ""
        out["k_repeat_ratio"] = clean_float(sum(pairs[i]["key"] == pairs[i-1]["key"] for i in range(1, n)) / (n - 1)) if n > 1 else ""
        out["k_unique_category_count"] = len(categories)
        for cat in ("alphanumeric", "correction", "enter_tab", "modifier", "space", "navigation", "function", "other"):
            out[f"k_category_{cat}_ratio"] = clean_float(categories[cat] / n) if n else ""
        out["k_correction_count"] = categories["correction"]
        return out

    def mouse_features(self, strokes: Sequence[dict], window_start: float, window_end: float) -> dict:
        out: dict[str, object] = {}
        n = len(strokes)
        dt = [s["dt"] for s in strokes]
        dist = [s["distance"] for s in strokes]
        vel = [s["velocity"] for s in strokes]
        duration = max(window_end - window_start, 1e-9)
        out["m_stroke_count"] = n
        out["m_window_span_s"] = clean_float(window_end - window_start)
        out["m_stroke_rate_s"] = clean_float(n / duration)
        out.update(stats(dt, "m_dt_ms", 1000.0))
        out.update(stats(dist, "m_distance_px", 1.0))
        out.update(stats(vel, "m_velocity_px_s", 1.0))
        out["m_path_length_px"] = clean_float(sum(dist))
        if strokes:
            net = math.hypot(strokes[-1]["x1"] - strokes[0]["x0"], strokes[-1]["y1"] - strokes[0]["y0"])
            out["m_net_displacement_px"] = clean_float(net)
            out["m_movement_efficiency"] = clean_float(net / sum(dist)) if sum(dist) > 1e-12 else ""
        else:
            out["m_net_displacement_px"] = ""
            out["m_movement_efficiency"] = ""
        out["m_zero_movement_ratio"] = clean_float(sum(x <= 1e-12 for x in dist) / n) if n else ""
        out["m_idle_500ms_ratio"] = clean_float(sum(x > 0.5 for x in dt) / n) if n else ""
        out["m_idle_1s_ratio"] = clean_float(sum(x > 1.0 for x in dt) / n) if n else ""
        out["m_idle_2s_ratio"] = clean_float(sum(x > 2.0 for x in dt) / n) if n else ""

        accelerations: list[float] = []
        accel_times: list[float] = []
        for a, b in zip(strokes, strokes[1:]):
            dtime = b["end_ts"] - a["end_ts"]
            if dtime > 0:
                accelerations.append((b["velocity"] - a["velocity"]) / dtime)
                accel_times.append(b["end_ts"])
        jerks: list[float] = []
        for i in range(1, len(accelerations)):
            dtime = accel_times[i] - accel_times[i-1]
            if dtime > 0:
                jerks.append((accelerations[i] - accelerations[i-1]) / dtime)
        out.update(stats(accelerations, "m_acceleration_px_s2", 1.0))
        out.update(stats([abs(x) for x in accelerations], "m_abs_acceleration_px_s2", 1.0))
        out.update(stats(jerks, "m_jerk_px_s3", 1.0))
        out.update(stats([abs(x) for x in jerks], "m_abs_jerk_px_s3", 1.0))

        turns: list[float] = []
        angles = [s["angle"] for s in strokes]
        for a, b in zip(angles, angles[1:]):
            if a is not None and b is not None:
                turns.append(abs(angle_difference(a, b)))
        out.update(stats(turns, "m_turn_angle_rad", 1.0))
        out["m_direction_change_45deg_ratio"] = clean_float(sum(x > math.pi/4 for x in turns) / len(turns)) if turns else ""
        out["m_direction_change_90deg_ratio"] = clean_float(sum(x > math.pi/2 for x in turns) / len(turns)) if turns else ""

        event_counts = Counter()
        if strokes:
            event_counts[strokes[0]["event_a"]] += 1
            for s in strokes:
                event_counts[s["event_b"]] += 1
        event_total = sum(event_counts.values())
        for ev in ("mousemove", "mouseout", "mouseover", "mousedown", "mouseup", "mousewheel"):
            out[f"m_event_{ev}_ratio"] = clean_float(event_counts[ev] / event_total) if event_total else ""
        out["m_button_event_ratio"] = clean_float((event_counts["mousedown"] + event_counts["mouseup"]) / event_total) if event_total else ""
        out["m_wheel_event_ratio"] = clean_float(event_counts["mousewheel"] / event_total) if event_total else ""
        return out

    @staticmethod
    def keyboard_sequence_steps(pairs: Sequence[dict], window_start: float) -> list[dict]:
        categories = ("alphanumeric", "correction", "enter_tab", "modifier", "space", "navigation", "function", "other")
        steps: list[dict] = []
        for i, pair in enumerate(pairs):
            pp = pair["down_ts"] - pairs[i - 1]["down_ts"] if i > 0 else 0.0
            rp = pair["down_ts"] - pairs[i - 1]["up_ts"] if i > 0 else 0.0
            row = {
                "t_rel_s": clean_float(pair["down_ts"] - window_start),
                "dwell_ms": clean_float(pair["dwell"] * 1000.0),
                "press_press_ms": clean_float(pp * 1000.0),
                "release_press_ms": clean_float(rp * 1000.0),
                "has_previous": int(i > 0),
                "overlap_flag": int(i > 0 and rp < 0),
            }
            for cat in categories:
                row[f"cat_{cat}"] = int(pair["category"] == cat)
            steps.append(row)
        return steps

    @staticmethod
    def mouse_sequence_steps(strokes: Sequence[dict], window_start: float) -> list[dict]:
        events = ("mousemove", "mouseout", "mouseover", "mousedown", "mouseup", "mousewheel")
        steps: list[dict] = []
        prev_v = None
        prev_end = None
        prev_acc = None
        prev_acc_t = None
        for stroke in strokes:
            acceleration = 0.0
            has_acc = 0
            if prev_v is not None and prev_end is not None and stroke["end_ts"] > prev_end:
                acceleration = (stroke["velocity"] - prev_v) / (stroke["end_ts"] - prev_end)
                has_acc = 1
            jerk = 0.0
            has_jerk = 0
            if has_acc and prev_acc is not None and prev_acc_t is not None and stroke["end_ts"] > prev_acc_t:
                jerk = (acceleration - prev_acc) / (stroke["end_ts"] - prev_acc_t)
                has_jerk = 1
            angle = stroke["angle"]
            row = {
                "t_rel_s": clean_float(stroke["end_ts"] - window_start),
                "dt_ms": clean_float(stroke["dt"] * 1000.0),
                "dx_px": clean_float(stroke["dx"]),
                "dy_px": clean_float(stroke["dy"]),
                "distance_px": clean_float(stroke["distance"]),
                "velocity_px_s": clean_float(stroke["velocity"]),
                "angle_sin": clean_float(math.sin(angle)) if angle is not None else 0.0,
                "angle_cos": clean_float(math.cos(angle)) if angle is not None else 0.0,
                "zero_movement": int(stroke["distance"] <= 1e-12),
                "acceleration_px_s2": clean_float(acceleration),
                "has_acceleration": has_acc,
                "jerk_px_s3": clean_float(jerk),
                "has_jerk": has_jerk,
            }
            for ev in events:
                row[f"event_a_{ev}"] = int(stroke["event_a"] == ev)
                row[f"event_b_{ev}"] = int(stroke["event_b"] == ev)
            steps.append(row)
            prev_v = stroke["velocity"]
            prev_end = stroke["end_ts"]
            if has_acc:
                prev_acc = acceleration
                prev_acc_t = stroke["end_ts"]
        return steps

    def build_base_metadata(self, *, family: str, config: str, split: str, user: str, session_id: str,
                            segment_id: str, order: int, start: float, end: float,
                            source_start: int | str, source_end: int | str,
                            k_count: int, m_count: int, k_min: int, m_min: int) -> dict:
        k_obs, m_obs = int(k_count > 0), int(m_count > 0)
        k_ok, m_ok = int(k_count >= k_min), int(m_count >= m_min)
        win_id = stable_id((family, config, split, user, session_id, segment_id, order, f"{start:.6f}", f"{end:.6f}", source_start, source_end),
                           {"keyboard": "KW", "mouse": "MW", "multimodal": "TW"}[family])
        return {
            "window_id": win_id, "window_family": family, "window_config": config,
            "split": split, "user_id": user, "session_id": session_id,
            "segment_id": segment_id, "window_order": order,
            "window_start_time": fmt_epoch(start), "window_end_time": fmt_epoch(end),
            "window_start_epoch": clean_float(start, 6), "window_end_epoch": clean_float(end, 6),
            "source_start_index": source_start, "source_end_index": source_end,
            "keyboard_observed": k_obs, "mouse_observed": m_obs,
            "keyboard_eligible": k_ok, "mouse_eligible": m_ok,
            "eligible_any": int(bool(k_ok or m_ok)), "eligible_both": int(bool(k_ok and m_ok)),
            "primary_multimodal": int(family == "multimodal" and k_ok and m_ok),
        }

    def add_window(self, family: str, row: dict) -> None:
        split = row["split"]
        self.feature_rows[family][split].append(row)
        self.window_manifest.append({k: row.get(k, "") for k in METADATA_COLUMNS})

    def materialize_segment(self, seg: dict, events: list[dict]) -> dict:
        pairs = self.pair_keyboard(events)
        strokes = self.mouse_strokes(events)
        split, user, sid, segid = seg["split"], seg["user"], seg["session_id"], seg["segment_id"]
        counts = Counter()

        # Keyboard fixed windows: 5 pairs, step 2 (frozen Stage 2A behavior).
        order = 0
        for start_idx in range(0, max(0, len(pairs) - KEY_WINDOW_SIZE + 1), KEY_WINDOW_STEP):
            chunk = pairs[start_idx:start_idx + KEY_WINDOW_SIZE]
            if len(chunk) < KEY_WINDOW_SIZE:
                continue
            order += 1
            wstart, wend = chunk[0]["down_ts"], max(p["up_ts"] for p in chunk)
            meta = self.build_base_metadata(
                family="keyboard", config="keyboard_fixed_5_pairs_stage2a_step2",
                split=split, user=user, session_id=sid, segment_id=segid, order=order,
                start=wstart, end=wend, source_start=start_idx, source_end=start_idx + KEY_WINDOW_SIZE - 1,
                k_count=len(chunk), m_count=0, k_min=KEY_WINDOW_SIZE, m_min=10**9,
            )
            row = {**meta, **self.keyboard_features(chunk, wstart, wend)}
            self.add_window("keyboard", row)
            self.sequence_records["keyboard"][split].append({
                "window_id": meta["window_id"], "split": split, "user_id": user,
                "session_id": sid, "segment_id": segid, "steps": self.keyboard_sequence_steps(chunk, wstart),
            })
            counts["keyboard"] += 1

        # Mouse fixed windows: 10 strokes, step 5.
        order = 0
        for start_idx in range(0, max(0, len(strokes) - MOUSE_WINDOW_SIZE + 1), MOUSE_WINDOW_STEP):
            chunk = strokes[start_idx:start_idx + MOUSE_WINDOW_SIZE]
            if len(chunk) < MOUSE_WINDOW_SIZE:
                continue
            order += 1
            wstart, wend = chunk[0]["start_ts"], chunk[-1]["end_ts"]
            meta = self.build_base_metadata(
                family="mouse", config="mouse_fixed_10_strokes_50pct",
                split=split, user=user, session_id=sid, segment_id=segid, order=order,
                start=wstart, end=wend, source_start=start_idx, source_end=start_idx + MOUSE_WINDOW_SIZE - 1,
                k_count=0, m_count=len(chunk), k_min=10**9, m_min=MOUSE_WINDOW_SIZE,
            )
            row = {**meta, **self.mouse_features(chunk, wstart, wend)}
            self.add_window("mouse", row)
            self.sequence_records["mouse"][split].append({
                "window_id": meta["window_id"], "split": split, "user_id": user,
                "session_id": sid, "segment_id": segid, "steps": self.mouse_sequence_steps(chunk, wstart),
            })
            counts["mouse"] += 1

        # Synchronized 10-second windows, 5-second step. Materialize eligible_any;
        # eligible_both marks the prespecified primary multimodal subset.
        if events and seg["end_epoch"] - seg["start_epoch"] + 1e-9 >= TIME_WINDOW_SECONDS:
            pair_times = [p["down_ts"] for p in pairs]
            stroke_times = [s["end_ts"] for s in strokes]
            wstart = seg["start_epoch"]
            order = 0
            while wstart + TIME_WINDOW_SECONDS <= seg["end_epoch"] + 1e-9:
                wend = wstart + TIME_WINDOW_SECONDS
                k0, k1 = bisect.bisect_left(pair_times, wstart), bisect.bisect_left(pair_times, wend)
                m0, m1 = bisect.bisect_left(stroke_times, wstart), bisect.bisect_left(stroke_times, wend)
                kp = pairs[k0:k1]
                ms = strokes[m0:m1]
                k_ok, m_ok = len(kp) >= TIME_MIN_KEY, len(ms) >= TIME_MIN_MOUSE
                if k_ok or m_ok:
                    order += 1
                    meta = self.build_base_metadata(
                        family="multimodal", config="time_10s_50pct_minK3_minM3",
                        split=split, user=user, session_id=sid, segment_id=segid, order=order,
                        start=wstart, end=wend, source_start=f"K{k0};M{m0}", source_end=f"K{k1-1};M{m1-1}",
                        k_count=len(kp), m_count=len(ms), k_min=TIME_MIN_KEY, m_min=TIME_MIN_MOUSE,
                    )
                    row = dict(meta)
                    row.update(self.keyboard_features(kp, wstart, wend) if kp else self.keyboard_features([], wstart, wend))
                    row.update(self.mouse_features(ms, wstart, wend) if ms else self.mouse_features([], wstart, wend))
                    self.add_window("multimodal", row)
                    self.sequence_records["multimodal"][split].append({
                        "window_id": meta["window_id"], "split": split, "user_id": user,
                        "session_id": sid, "segment_id": segid,
                        "keyboard_eligible": meta["keyboard_eligible"], "mouse_eligible": meta["mouse_eligible"],
                        "eligible_both": meta["eligible_both"],
                        "keyboard_steps": self.keyboard_sequence_steps(kp, wstart),
                        "mouse_steps": self.mouse_sequence_steps(ms, wstart),
                    })
                    counts["multimodal_any"] += 1
                    if k_ok and m_ok:
                        counts["multimodal_both"] += 1
                wstart += TIME_WINDOW_STEP_SECONDS
        return dict(counts)

    def materialize_all(self) -> dict:
        per_segment_rows: list[dict] = []
        total = Counter()
        for i, seg in enumerate(self.segments, start=1):
            events = self.fetch_segment_events(seg)
            counts = self.materialize_segment(seg, events)
            total.update(counts)
            per_segment_rows.append({
                "user": seg["user"], "session_id": seg["session_id"], "segment_id": seg["segment_id"],
                "split": seg["split"], "event_count": len(events),
                "keyboard_windows": counts.get("keyboard", 0), "mouse_windows": counts.get("mouse", 0),
                "multimodal_eligible_any_windows": counts.get("multimodal_any", 0),
                "multimodal_eligible_both_windows": counts.get("multimodal_both", 0),
            })
            if i % 100 == 0:
                self.log(f"Processed {i}/{len(self.segments)} frozen core segments")
        write_csv(self.output_dir / "window_counts_by_segment.csv", per_segment_rows,
                  ["user", "session_id", "segment_id", "split", "event_count", "keyboard_windows", "mouse_windows",
                   "multimodal_eligible_any_windows", "multimodal_eligible_both_windows"])
        self.log(f"Materialized windows: {dict(total)}")
        return dict(total)

    def write_sequence_files(self) -> dict:
        counts: dict[str, int] = {}
        manifest_rows: list[dict] = []
        for family in ("keyboard", "mouse", "multimodal"):
            for split in SPLITS:
                records = self.sequence_records[family][split]
                records.sort(key=lambda r: (user_sort_key(r["user_id"]), r["session_id"], r["segment_id"], r["window_id"]))
                path = self.output_dir / f"{family}_sequences_{split}.jsonl"
                with path.open("w", encoding="utf-8", newline="") as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
                key = f"{family}_{split}"
                counts[key] = len(records)
                manifest_rows.append({
                    "family": family, "split": split, "file": path.name,
                    "records": len(records), "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
        write_csv(self.output_dir / "sequence_file_manifest.csv", manifest_rows,
                  ["family", "split", "file", "records", "size_bytes", "sha256"])
        schema = """# Sequence Representation Schema

## Keyboard stream

Each fixed keyboard window contains exactly five numeric steps. Each multimodal keyboard stream is variable length. A step contains relative time, dwell time, press–press latency, release–press latency, previous-step and overlap masks, and one-hot coarse key categories. Raw key codes and typed content are not exported.

## Mouse stream

Each fixed mouse window contains exactly ten numeric steps. Each multimodal mouse stream is variable length. A step contains relative time, elapsed time, displacement, distance, velocity, sine/cosine direction, zero-movement flag, acceleration/jerk with validity masks, and one-hot source/destination event types.

## Padding policy

No padding is performed in Stage 2B. Later tensor construction must derive padding/truncation lengths from Training only, store the masks, and apply the frozen policy unchanged to Validation and Test.
"""
        (self.output_dir / "sequence_representation_schema.md").write_text(schema, encoding="utf-8")
        return counts

    def write_feature_tables(self) -> dict:
        counts = {}
        for family in ("keyboard", "mouse", "multimodal"):
            for split in SPLITS:
                rows = self.feature_rows[family][split]
                rows.sort(key=lambda r: (user_sort_key(r["user_id"]), r["session_id"], r["segment_id"], int(r["window_order"]), r["window_id"]))
                path = self.output_dir / f"{family}_features_{split}.csv"
                if rows:
                    feature_cols = [c for c in rows[0].keys() if c not in METADATA_COLUMNS]
                    fields = METADATA_COLUMNS + feature_cols
                else:
                    fields = METADATA_COLUMNS
                write_csv(path, rows, fields)
                counts[f"{family}_{split}"] = len(rows)
        self.window_manifest.sort(key=lambda r: (r["window_family"], r["split"], user_sort_key(r["user_id"]), r["session_id"], r["segment_id"], int(r["window_order"])))
        write_csv(self.output_dir / "window_manifest.csv", self.window_manifest, METADATA_COLUMNS)
        primary = [
            {"window_id": r["window_id"], "split": r["split"], "user_id": r["user_id"], "session_id": r["session_id"], "segment_id": r["segment_id"]}
            for r in self.window_manifest if r["window_family"] == "multimodal" and int(r["primary_multimodal"]) == 1
        ]
        write_csv(self.output_dir / "multimodal_primary_eligible_both_index.csv", primary,
                  ["window_id", "split", "user_id", "session_id", "segment_id"])
        return counts

    def build_feature_dictionary(self) -> list[dict]:
        rows: list[dict] = []
        for col in METADATA_COLUMNS:
            role = "mask" if col in {"keyboard_observed", "mouse_observed", "keyboard_eligible", "mouse_eligible", "eligible_any", "eligible_both", "primary_multimodal"} else "metadata"
            rows.append({
                "feature_name": col, "family": "all", "role": role,
                "dtype": "integer" if role == "mask" or col in {"window_order", "source_start_index", "source_end_index"} else "string_or_numeric_metadata",
                "unit": "", "definition": "Lineage/mask field; retained for audit and excluded from learned feature matrix unless explicitly marked as a modality mask.",
                "source": "Frozen Stage 2A manifest and deterministic Stage 2B window construction",
                "missing_policy": "must_not_be_missing" if col not in {"source_start_index", "source_end_index"} else "allowed",
                "model_input_policy": "mask_allowed" if role == "mask" else "prohibited_identifier_or_lineage",
            })
        sample_rows = []
        for family in ("keyboard", "mouse", "multimodal"):
            for split in SPLITS:
                if self.feature_rows[family][split]:
                    sample_rows.append((family, self.feature_rows[family][split][0]))
                    break
        seen = set(METADATA_COLUMNS)
        for family, sample in sample_rows:
            for name in sample:
                if name in seen:
                    continue
                seen.add(name)
                if name.startswith("k_"):
                    fam = "keyboard"
                    source = "Valid keydown/keyup pairs within one frozen segment/window"
                elif name.startswith("m_"):
                    fam = "mouse"
                    source = "Consecutive positive-dt mouse strokes within one frozen segment/window"
                else:
                    fam = family
                    source = "Derived window statistics"
                unit = ""
                if name.endswith("_ms") or "_ms_" in name:
                    unit = "milliseconds"
                elif name.endswith("_s") or "_s_" in name:
                    unit = "seconds_or_rate_per_second"
                elif "_px" in name:
                    unit = "pixels_or_pixel_rate"
                elif "ratio" in name or "efficiency" in name or name.endswith("_cv"):
                    unit = "unitless"
                rows.append({
                    "feature_name": name, "family": fam, "role": "numeric_feature",
                    "dtype": "float_or_integer", "unit": unit,
                    "definition": self.describe_feature(name), "source": source,
                    "missing_policy": "empty_when_statistic_undefined; imputation_must_be_fit_on_training_only",
                    "model_input_policy": "allowed_after_training_only_preprocessing",
                })
        write_csv(self.output_dir / "feature_dictionary.csv", rows,
                  ["feature_name", "family", "role", "dtype", "unit", "definition", "source", "missing_policy", "model_input_policy"])
        provenance = []
        for r in rows:
            if r["role"] != "numeric_feature":
                continue
            provenance.append({
                "feature_name": r["feature_name"], "family": r["family"],
                "source_event_scope": r["source"],
                "transformation": r["definition"],
                "boundary_constraint": "within_one_frozen_session_and_60s_idle_segment",
                "learned_parameter": "none",
                "training_only_requirement": "Any later imputation/scaling/selection must be fitted on Training only.",
                "identifier_exposure": "none_raw_identifiers_or_raw_key_codes",
            })
        write_csv(self.output_dir / "feature_provenance.csv", provenance,
                  ["feature_name", "family", "source_event_scope", "transformation", "boundary_constraint",
                   "learned_parameter", "training_only_requirement", "identifier_exposure"])
        return rows

    @staticmethod
    def describe_feature(name: str) -> str:
        fixed = {
            "k_pair_count": "Number of valid keydown/keyup pairs in the window.",
            "k_window_span_s": "Window temporal span used for keyboard feature rates.",
            "k_typing_rate_pairs_s": "Valid key pairs per second.",
            "k_typing_rate_pairs_min": "Valid key pairs per minute.",
            "k_overlap_ratio": "Fraction of release-to-next-press latencies below zero, indicating overlapping keystrokes.",
            "k_pause_500ms_ratio": "Fraction of nonnegative press-to-press intervals longer than 0.5 seconds.",
            "k_pause_1s_ratio": "Fraction of nonnegative press-to-press intervals longer than 1 second.",
            "k_pause_2s_ratio": "Fraction of nonnegative press-to-press intervals longer than 2 seconds.",
            "k_burst_count_1s": "Number of typing bursts separated by press-to-press gaps greater than 1 second.",
            "k_burst_mean_len_1s": "Mean number of key pairs per 1-second-gap-defined burst.",
            "k_burst_max_len_1s": "Maximum number of key pairs in a 1-second-gap-defined burst.",
            "k_repeat_ratio": "Fraction of consecutive valid pairs with the same raw key code; raw codes themselves are not exported.",
            "k_unique_category_count": "Number of coarse key categories observed.",
            "k_correction_count": "Count of Backspace/Delete-category key pairs.",
            "m_stroke_count": "Number of valid consecutive mouse-event strokes.",
            "m_window_span_s": "Window temporal span used for mouse feature rates.",
            "m_stroke_rate_s": "Valid mouse strokes per second.",
            "m_path_length_px": "Sum of Euclidean stroke distances.",
            "m_net_displacement_px": "Distance from the first stroke start to the last stroke end.",
            "m_movement_efficiency": "Net displacement divided by path length.",
            "m_zero_movement_ratio": "Fraction of valid strokes with zero coordinate displacement.",
            "m_idle_500ms_ratio": "Fraction of stroke elapsed times longer than 0.5 seconds.",
            "m_idle_1s_ratio": "Fraction of stroke elapsed times longer than 1 second.",
            "m_idle_2s_ratio": "Fraction of stroke elapsed times longer than 2 seconds.",
            "m_direction_change_45deg_ratio": "Fraction of valid consecutive directions changing by more than 45 degrees.",
            "m_direction_change_90deg_ratio": "Fraction of valid consecutive directions changing by more than 90 degrees.",
            "m_button_event_ratio": "Fraction of reconstructed mouse events that are mousedown or mouseup.",
            "m_wheel_event_ratio": "Fraction of reconstructed mouse events that are mousewheel.",
        }
        if name in fixed:
            return fixed[name]
        if name.startswith("k_category_"):
            return "Fraction of key pairs assigned to the named coarse, non-content key category."
        if name.startswith("m_event_"):
            return "Fraction of reconstructed mouse events of the named event type."
        if any(token in name for token in ("_mean", "_std", "_median", "_p10", "_p90", "_p95", "_min", "_max", "_iqr", "_cv")):
            return "Distribution summary of the quantity named in the feature prefix."
        return "Deterministic, non-learned behavioral feature derived within one frozen window."

    def quality_audit(self) -> dict:
        quality_rows: list[dict] = []
        bounds_rows: list[dict] = []
        outlier_rows: list[dict] = []
        feature_names_by_family: dict[str, list[str]] = {}

        for family in ("keyboard", "mouse", "multimodal"):
            sample = next((self.feature_rows[family][s][0] for s in SPLITS if self.feature_rows[family][s]), None)
            if sample is None:
                continue
            feature_names = [c for c in sample.keys() if c not in METADATA_COLUMNS]
            feature_names_by_family[family] = feature_names
            train_values: dict[str, list[float]] = defaultdict(list)
            for name in feature_names:
                for row in self.feature_rows[family]["train"]:
                    val = row.get(name, "")
                    try:
                        x = float(val)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(x):
                        train_values[name].append(x)
            bounds: dict[str, tuple[float | None, float | None, float | None, float | None]] = {}
            for name in feature_names:
                vals = train_values[name]
                q1, q3 = percentile(vals, .25), percentile(vals, .75)
                if q1 is None or q3 is None:
                    lower = upper = None
                else:
                    iqr = q3 - q1
                    lower, upper = q1 - 3.0 * iqr, q3 + 3.0 * iqr
                bounds[name] = (q1, q3, lower, upper)
                bounds_rows.append({
                    "family": family, "feature_name": name, "training_nonmissing": len(vals),
                    "training_q1": clean_float(q1), "training_q3": clean_float(q3),
                    "training_iqr": clean_float((q3 - q1) if q1 is not None and q3 is not None else None),
                    "diagnostic_lower_3iqr": clean_float(lower), "diagnostic_upper_3iqr": clean_float(upper),
                    "action": "audit_only_no_clipping",
                })
            for split in SPLITS:
                rows = self.feature_rows[family][split]
                for name in feature_names:
                    vals: list[float] = []
                    missing = nonfinite = 0
                    for row in rows:
                        val = row.get(name, "")
                        if val == "" or val is None:
                            missing += 1
                            continue
                        try:
                            x = float(val)
                        except (TypeError, ValueError):
                            missing += 1
                            continue
                        if not math.isfinite(x):
                            nonfinite += 1
                            continue
                        vals.append(x)
                    quality_rows.append({
                        "family": family, "split": split, "feature_name": name,
                        "rows": len(rows), "nonmissing": len(vals), "missing": missing,
                        "missing_rate": clean_float(missing / len(rows)) if rows else "",
                        "nonfinite": nonfinite, "min": clean_float(min(vals)) if vals else "",
                        "max": clean_float(max(vals)) if vals else "", "mean": clean_float(statistics.fmean(vals)) if vals else "",
                        "std": clean_float(statistics.pstdev(vals)) if len(vals) > 1 else (0.0 if vals else ""),
                    })
                    q1, q3, lower, upper = bounds[name]
                    low_n = sum(x < lower for x in vals) if lower is not None else 0
                    high_n = sum(x > upper for x in vals) if upper is not None else 0
                    outlier_rows.append({
                        "family": family, "split": split, "feature_name": name,
                        "nonmissing": len(vals), "below_training_3iqr": low_n,
                        "above_training_3iqr": high_n,
                        "diagnostic_outlier_rate": clean_float((low_n + high_n) / len(vals)) if vals else "",
                        "note": "Training-derived diagnostic only; no clipping or removal performed.",
                    })
        write_csv(self.output_dir / "feature_quality_by_split.csv", quality_rows,
                  ["family", "split", "feature_name", "rows", "nonmissing", "missing", "missing_rate", "nonfinite", "min", "max", "mean", "std"])
        write_csv(self.output_dir / "training_only_feature_bounds.csv", bounds_rows,
                  ["family", "feature_name", "training_nonmissing", "training_q1", "training_q3", "training_iqr", "diagnostic_lower_3iqr", "diagnostic_upper_3iqr", "action"])
        write_csv(self.output_dir / "training_bound_outlier_audit.csv", outlier_rows,
                  ["family", "split", "feature_name", "nonmissing", "below_training_3iqr", "above_training_3iqr", "diagnostic_outlier_rate", "note"])

        scope_rows = []
        multi_sample = next((self.feature_rows["multimodal"][s][0] for s in SPLITS if self.feature_rows["multimodal"][s]), None)
        if multi_sample:
            multi_features = [c for c in multi_sample if c not in METADATA_COLUMNS]
            for split in SPLITS:
                all_rows = self.feature_rows["multimodal"][split]
                scopes = {
                    "eligible_any": all_rows,
                    "primary_eligible_both": [r for r in all_rows if int(r["primary_multimodal"]) == 1],
                }
                for scope, rows_scope in scopes.items():
                    for name in multi_features:
                        missing = 0
                        for row in rows_scope:
                            val = row.get(name, "")
                            if val in ("", None):
                                missing += 1
                        scope_rows.append({
                            "split": split, "scope": scope, "feature_name": name, "rows": len(rows_scope),
                            "missing": missing, "missing_rate": clean_float(missing / len(rows_scope)) if rows_scope else "",
                        })
        write_csv(self.output_dir / "multimodal_feature_missingness_by_scope.csv", scope_rows,
                  ["split", "scope", "feature_name", "rows", "missing", "missing_rate"])

        total_nonfinite = sum(int(r["nonfinite"]) for r in quality_rows)
        if total_nonfinite:
            self.fail("NONFINITE_FEATURES", f"Found {total_nonfinite} non-finite feature values")
        return {"quality_rows": len(quality_rows), "training_bounds": len(bounds_rows), "nonfinite_values": total_nonfinite}

    def leakage_and_coverage_audit(self) -> dict:
        ids = [r["window_id"] for r in self.window_manifest]
        duplicate_ids = len(ids) - len(set(ids))
        if duplicate_ids:
            self.fail("DUPLICATE_WINDOW_IDS", f"Detected {duplicate_ids} duplicate window IDs")

        session_splits: dict[str, set[str]] = defaultdict(set)
        segment_splits: dict[str, set[str]] = defaultdict(set)
        user_split_family = Counter()
        family_split = Counter()
        for r in self.window_manifest:
            session_splits[r["session_id"]].add(r["split"])
            segment_splits[r["segment_id"]].add(r["split"])
            user_split_family[(r["user_id"], r["split"], r["window_family"])] += 1
            family_split[(r["window_family"], r["split"])] += 1
        session_conflicts = {k: sorted(v) for k, v in session_splits.items() if len(v) > 1}
        segment_conflicts = {k: sorted(v) for k, v in segment_splits.items() if len(v) > 1}
        if session_conflicts:
            self.fail("SESSION_SPLIT_LEAKAGE", f"{len(session_conflicts)} sessions appear in multiple splits")
        if segment_conflicts:
            self.fail("SEGMENT_SPLIT_LEAKAGE", f"{len(segment_conflicts)} segments appear in multiple splits")

        coverage_rows = []
        zero_cells = []
        users = sorted({r["user"] for r in self.core_manifest.values()}, key=user_sort_key)
        for family in ("keyboard", "mouse", "multimodal"):
            for user in users:
                for split in SPLITS:
                    n = user_split_family[(user, split, family)]
                    primary_n = 0
                    if family == "multimodal":
                        primary_n = sum(1 for r in self.feature_rows[family][split] if r["user_id"] == user and int(r["primary_multimodal"]) == 1)
                    coverage_rows.append({
                        "family": family, "user_id": user, "split": split,
                        "materialized_windows": n, "primary_multimodal_both_windows": primary_n,
                    })
                    if n == 0:
                        zero_cells.append((family, user, split, "all_materialized"))
                    if family == "multimodal" and primary_n == 0:
                        zero_cells.append((family, user, split, "primary_eligible_both"))
        write_csv(self.output_dir / "window_counts_by_user_split.csv", coverage_rows,
                  ["family", "user_id", "split", "materialized_windows", "primary_multimodal_both_windows"])
        if zero_cells:
            for fam, user, split, scope in zero_cells:
                self.warn("ZERO_WINDOW_CELL", "high", f"No windows for {fam}/{user}/{split}/{scope}")

        forbidden_feature_names = {
            "user", "user_id", "session_id", "segment_id", "_id", "raw_id", "timestamp", "timestamp_raw",
            "ua", "screen", "screenh", "windoww", "windowh", "documentw", "documenth", "cursor",
            "source_row", "file_path",
        }
        accidental = []
        for family in ("keyboard", "mouse", "multimodal"):
            sample = next((self.feature_rows[family][s][0] for s in SPLITS if self.feature_rows[family][s]), {})
            model_candidates = set(sample) - set(METADATA_COLUMNS)
            accidental.extend(sorted(model_candidates & forbidden_feature_names))
        if accidental:
            self.fail("FORBIDDEN_FEATURE_EXPORTED", f"Forbidden identifiers found among numeric feature candidates: {sorted(set(accidental))}")

        audit_rows = [
            {"check": "frozen_manifest_sha256", "status": "PASS", "detail": EXPECTED_FROZEN_HASH},
            {"check": "unique_window_ids", "status": "PASS" if duplicate_ids == 0 else "FAIL", "detail": str(duplicate_ids)},
            {"check": "session_in_single_split", "status": "PASS" if not session_conflicts else "FAIL", "detail": str(len(session_conflicts))},
            {"check": "segment_in_single_split", "status": "PASS" if not segment_conflicts else "FAIL", "detail": str(len(segment_conflicts))},
            {"check": "no_forbidden_numeric_features", "status": "PASS" if not accidental else "FAIL", "detail": ";".join(accidental)},
            {"check": "normalization_or_feature_selection", "status": "NOT_PERFORMED", "detail": "Stage 2B exports raw deterministic features only."},
            {"check": "claims_or_models", "status": "NOT_PERFORMED", "detail": "No genuine/impostor claims, calibration, thresholds, or models."},
        ]
        write_csv(self.output_dir / "leakage_integrity_checks.csv", audit_rows, ["check", "status", "detail"])
        return {
            "window_ids": len(ids), "duplicate_window_ids": duplicate_ids,
            "session_split_conflicts": len(session_conflicts), "segment_split_conflicts": len(segment_conflicts),
            "zero_coverage_cells": len(zero_cells), "forbidden_feature_hits": len(accidental),
            "family_split_counts": {f"{k[0]}_{k[1]}": v for k, v in sorted(family_split.items())},
        }

    def write_reports(self, frozen_stats: dict, import_stats: dict, materialized: dict,
                      table_counts: dict, sequence_counts: dict, quality_stats: dict, leakage_stats: dict) -> None:
        warnings_failures = self.warnings + self.failures
        write_csv(self.output_dir / "warnings_and_failures.csv", warnings_failures,
                  ["type", "code", "severity", "message", "context"])

        report = f"""# {STAGE_NAME} — Leakage Control Report

## Binding inputs

- Frozen session manifest SHA-256: `{EXPECTED_FROZEN_HASH}`
- Frozen 60-second segment manifest SHA-256: `{EXPECTED_SEGMENT_HASH}`
- Frozen semantic duplicate list SHA-256: `{EXPECTED_DUPLICATE_HASH}`
- Core enrolled sessions: {len(self.core_manifest)}
- Core derived segments: {len(self.segments)}

## Materialized representations

1. Keyboard-only: 5 valid key pairs, deterministic step of 2 pairs. The legacy Stage 2A label says 50% overlap; because the window size is odd, this exact implementation has an effective 60% pair overlap and is explicitly documented rather than silently changed.
2. Mouse-only: 10 valid consecutive mouse strokes, step 5 strokes (50% overlap).
3. Multimodal: 10-second windows, 5-second step, within one frozen idle segment only. All `eligible_any` windows are exported with modality masks; `eligible_both` identifies the prespecified primary subset requiring at least 3 key pairs and 3 mouse strokes.

## Leakage controls

- No original session or derived segment appears in more than one split.
- No window crosses an original-session or 60-second-idle segment boundary.
- Raw identifiers, browser/device fields, raw timestamps, raw key codes, source rows, and file paths are not numeric model features.
- Raw key codes are reduced only to coarse categories; raw codes are not exported.
- Feature extraction is deterministic and occurs after the frozen session split.
- No scaler, imputer, encoder, feature selector, dimensionality reduction, calibration, threshold, claim generator, or model is fitted in Stage 2B.
- Training-derived 3×IQR bounds are diagnostic only; no row or value is clipped or removed.
- Missing values remain empty when a statistic is mathematically undefined; any later imputation must be fitted on Training only.

## Status

- Warnings: {len(self.warnings)}
- Failures: {len(self.failures)}
- Decision: {'FAIL' if self.failures else 'PASS_WITH_WARNINGS' if self.warnings else 'PASS'}
"""
        (self.output_dir / "stage2b_leakage_control_report.md").write_text(report, encoding="utf-8")

        policy = """# Excluded Feature Policy

The following fields are lineage or identifiers and must never be supplied to a model as behavioral predictors:

`user_id`, `session_id`, `segment_id`, `window_id`, raw timestamps, `_id`, source-row indices, user-agent, screen/window/document dimensions, cursor, file path, and raw key codes.

The split label is orchestration metadata only. Modality masks may be used only to distinguish observed/missing modalities in multimodal experiments. Coarse key-category ratios are permitted because they do not expose the raw key sequence.

No normalization, missing-value imputation, feature selection, or clipping is performed here. Any such transformation in a later stage must be fitted on Training only, stored, and applied unchanged to Validation and Test.
"""
        (self.output_dir / "excluded_feature_policy.md").write_text(policy, encoding="utf-8")

        recommendation = """# Recommended Next Step after Stage 2B Review

Do not train models yet. First review the local Stage 2B output and freeze the feature/sequence schemas. The next controlled step should fit missing-value handling, scaling, optional training-only feature pruning, and sequence padding lengths on Training only. Validation and Test must be transformed unchanged.

The primary multimodal experiment should use `primary_multimodal=1` (`eligible_both`). The broader `eligible_any` set should be retained as a prespecified robustness experiment using modality masks. Keyboard-only and mouse-only tables remain fair modality baselines.

No claim construction should occur until preprocessing artifacts and tensor schemas are frozen. Validation and Test claim-level scores must later retain `window_id`, `user_id`, `session_id`, `claimed_user_id`, `actual_user_id`, and `is_genuine`.
"""
        (self.output_dir / "recommended_next_stage.md").write_text(recommendation, encoding="utf-8")

        summary = {
            "stage": STAGE_NAME, "version": VERSION,
            "status": "FAIL" if self.failures else "PASS_WITH_WARNINGS" if self.warnings else "PASS",
            "started_at": self.log_lines[0].split("] ", 1)[0].lstrip("[") if self.log_lines else "",
            "finished_at": now_text(), "elapsed_seconds": round(time.time() - self.start_clock, 3),
            "source": {
                "path": str(self.input_path),
                "sha256": sha256_file(self.input_path) if self.input_path.is_file() else "directory_input",
            },
            "frozen_inputs": frozen_stats,
            "event_import": import_stats,
            "parameters": {
                "keyboard_window_size_pairs": KEY_WINDOW_SIZE,
                "keyboard_step_pairs": KEY_WINDOW_STEP,
                "keyboard_effective_overlap": 1.0 - KEY_WINDOW_STEP / KEY_WINDOW_SIZE,
                "mouse_window_size_strokes": MOUSE_WINDOW_SIZE,
                "mouse_step_strokes": MOUSE_WINDOW_STEP,
                "time_window_seconds": TIME_WINDOW_SECONDS,
                "time_step_seconds": TIME_WINDOW_STEP_SECONDS,
                "time_min_key_pairs": TIME_MIN_KEY,
                "time_min_mouse_strokes": TIME_MIN_MOUSE,
                "max_dwell_seconds": MAX_DWELL_SECONDS,
            },
            "materialized": materialized, "feature_table_counts": table_counts,
            "sequence_record_counts": sequence_counts,
            "feature_quality": quality_stats, "leakage_integrity": leakage_stats,
            "warning_count": len(self.warnings), "failure_count": len(self.failures),
        }
        write_json(self.output_dir / "stage2b_summary.json", summary)

    def create_output_manifest(self) -> None:
        rows = []
        for p in sorted(self.output_dir.rglob("*")):
            if not p.is_file() or p.name == "output_manifest_sha256.csv":
                continue
            rows.append({
                "relative_path": p.relative_to(self.output_dir).as_posix(),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
        write_csv(self.output_dir / "output_manifest_sha256.csv", rows,
                  ["relative_path", "size_bytes", "sha256"])

    def zip_output(self) -> Path:
        zip_path = self.output_dir.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for p in sorted(self.output_dir.rglob("*")):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(self.output_dir.parent).as_posix())
        return zip_path

    def cleanup(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        if self.sqlite_path and self.sqlite_path.exists():
            try:
                self.sqlite_path.unlink()
            except OSError:
                pass

    def run(self) -> int:
        try:
            self.prepare_output()
            frozen_stats = self.load_and_verify_frozen_inputs()
            import_stats = self.import_events()
            materialized = self.materialize_all()
            table_counts = self.write_feature_tables()
            sequence_counts = self.write_sequence_files()
            self.build_feature_dictionary()
            quality_stats = self.quality_audit()
            leakage_stats = self.leakage_and_coverage_audit()
            self.write_reports(frozen_stats, import_stats, materialized, table_counts, sequence_counts, quality_stats, leakage_stats)
            if self.failures:
                self.log(f"Completed with {len(self.failures)} failure(s).")
            else:
                self.log(f"Completed successfully with {len(self.warnings)} warning(s).")
            (self.output_dir / "execution_log.txt").write_text("\n".join(self.log_lines) + "\n", encoding="utf-8")
            self.create_output_manifest()
            zip_path = self.zip_output()
            print(f"Review ZIP: {zip_path}", flush=True)
            return 2 if self.failures else 0
        except Exception as exc:
            self.fail("UNHANDLED_EXCEPTION", str(exc))
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                write_csv(self.output_dir / "warnings_and_failures.csv", self.warnings + self.failures,
                          ["type", "code", "severity", "message", "context"])
                (self.output_dir / "execution_log.txt").write_text("\n".join(self.log_lines) + "\n", encoding="utf-8")
                self.create_output_manifest()
                self.zip_output()
            except Exception:
                pass
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        finally:
            self.cleanup()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=STAGE_NAME)
    p.add_argument("--input", required=True, help="Original UEBA dataset ZIP or extracted directory")
    p.add_argument("--output", required=True, help="Output review directory; a sibling ZIP is also created")
    p.add_argument("--frozen-inputs", default="", help="Optional Stage 2A review ZIP/directory. Default: bundled frozen inputs")
    p.add_argument("--batch-size", type=int, default=5000, help="SQLite streaming insert batch size")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    return p


def main() -> int:
    args = build_parser().parse_args()
    return Pilot(args).run()


if __name__ == "__main__":
    raise SystemExit(main())

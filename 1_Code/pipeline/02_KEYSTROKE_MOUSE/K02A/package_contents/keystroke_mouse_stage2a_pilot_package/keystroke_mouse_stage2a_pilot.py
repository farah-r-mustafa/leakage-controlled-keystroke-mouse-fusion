#!/usr/bin/env python3
"""
Keystroke–Mouse Stage 2A — Session Manifest, Temporal Repair and Window Feasibility Pilot

Read-only with respect to the source dataset. The program:
  * validates the frozen Stage-1 schema,
  * streams source rows into a temporary SQLite work database,
  * removes only exact semantic duplicates (never edits source files),
  * freezes chronological Train/Validation/Test session splits,
  * audits temporal ordering and idle-gap segmentation,
  * audits keydown/keyup pairing and consecutive mouse strokes,
  * estimates candidate window coverage,
  * performs no model training and creates no genuine/impostor claims.

Python: 3.12+
Dependencies: standard library only
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
import re
import shutil
import sqlite3
import statistics
import sys
import traceback
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Iterable, Iterator

VERSION = "2.2.0"
KEY_EVENTS = {"keydown", "keyup"}
MOUSE_EVENTS = {"mousemove", "mouseout", "mouseover", "mousedown", "mouseup", "mousewheel"}
EXPECTED_INFO = {"_id", "user", "session_id", "date", "ua", "screen", "screenh", "windoww", "windowh", "documentw", "documenth"}
EXPECTED_EVENTS = {"_id", "cursor", "timestamp", "xpos", "ypos", "key", "event", "user", "session_id"}
ENROLLED_USER_RE = re.compile(r"^User(\d+)$", re.IGNORECASE)
PREAUTH_NAME = "alluserspreauth"


@dataclass
class SourceTable:
    name: str
    opener: Callable[[], BinaryIO]
    headers: list[str]
    delimiter: str
    encoding: str
    kind: str


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    materialized = list(rows)
    if fieldnames is None:
        fieldnames = list(materialized[0].keys()) if materialized else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if materialized:
            w.writerows(materialized)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_dt(value: str) -> float | None:
    s = (value or "").strip()
    if not s:
        return None
    for candidate in (s, s.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate).timestamp()
        except ValueError:
            pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


def fmt_epoch(epoch: float | None) -> str:
    if epoch is None:
        return ""
    return datetime.fromtimestamp(epoch).isoformat(sep=" ", timespec="milliseconds")


def infer_delimiter(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()][:10]
    if not lines:
        return "\t"
    candidates = ["\t", ",", ";", "|"]
    scores = {d: sum(line.count(d) for line in lines) for d in candidates}
    if scores["\t"] >= max(scores.values()) and scores["\t"] > 0:
        return "\t"
    try:
        return csv.Sniffer().sniff("\n".join(lines), delimiters="\t,;|").delimiter
    except csv.Error:
        return max(scores, key=scores.get)


def read_header(opener: Callable[[], BinaryIO]) -> tuple[list[str], str, str]:
    raw = opener()
    try:
        sample = raw.read(65536)
    finally:
        raw.close()
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = sample.decode(encoding)
            delimiter = infer_delimiter(text)
            row = next(csv.reader(io.StringIO(text), delimiter=delimiter))
            return [x.strip() for x in row], delimiter, encoding
        except (UnicodeDecodeError, StopIteration, csv.Error):
            continue
    raise ValueError("Unable to decode or parse a delimited header")


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def sliding_count(n: int, size: int, step: int) -> int:
    if n < size:
        return 0
    return 1 + (n - size) // step


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def split_counts(n: int) -> tuple[int, int, int]:
    if n < 3:
        return n, 0, 0
    if n == 3:
        return 1, 1, 1
    if n == 4:
        return 2, 1, 1
    train = max(1, round_half_up(0.60 * n))
    val = max(1, round_half_up(0.20 * n))
    test = n - train - val
    if test < 1:
        need = 1 - test
        reducible_train = max(0, train - 1)
        take = min(need, reducible_train)
        train -= take
        need -= take
        if need:
            val = max(1, val - need)
        test = n - train - val
    return train, val, test


def user_sort_key(user: str) -> tuple[int, str]:
    m = ENROLLED_USER_RE.match(user)
    return (int(m.group(1)), user) if m else (10**9, user)


class Pilot:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.input_path = Path(args.input).expanduser().resolve()
        self.output_dir = Path(args.output).expanduser().resolve()
        self.output_zip = self.output_dir.with_suffix(".zip")
        self.work_db = self.output_dir / "stage2a_work.sqlite"
        self.logs: list[str] = []
        self.warnings: list[dict] = []
        self.failures: list[dict] = []
        self.info_table: SourceTable | None = None
        self.event_table: SourceTable | None = None
        self.conn: sqlite3.Connection | None = None
        self.session_rows: list[dict] = []
        self.primary_segments: list[dict] = []
        self.segment_event_cache: dict[tuple[str, int], list[dict]] = {}
        self.window_session_rows: list[dict] = []
        self.window_user_split_rows: list[dict] = []
        self.coverage_warnings: list[dict] = []
        self.idle_thresholds = sorted(set(float(x) for x in args.idle_thresholds.split(",") if x.strip()))
        self.primary_idle = float(args.primary_idle_seconds)
        if self.primary_idle not in self.idle_thresholds:
            self.idle_thresholds.append(self.primary_idle)
            self.idle_thresholds.sort()

    def log(self, text: str) -> None:
        self.logs.append(f"[{now_text()}] {text}")

    def warn(self, code: str, severity: str, message: str, session_id: str = "", user: str = "") -> None:
        self.warnings.append({
            "type": "warning", "code": code, "severity": severity,
            "user": user, "session_id": session_id, "message": message,
        })
        self.log(f"WARNING {severity} {code}: {message}")

    def fail(self, code: str, message: str) -> None:
        self.failures.append({
            "type": "failure", "code": code, "severity": "critical",
            "user": "", "session_id": "", "message": message,
        })
        self.log(f"FAILURE {code}: {message}")

    def prepare_output(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input does not exist: {self.input_path}")
        if self.output_dir == self.input_path or self.output_dir in self.input_path.parents:
            raise ValueError("Output cannot be the source path or inside a source directory")
        if self.output_dir.exists():
            if any(self.output_dir.iterdir()) and not self.args.overwrite:
                raise FileExistsError(
                    f"Output directory is not empty: {self.output_dir}. Use a new path or add --overwrite."
                )
            if self.args.overwrite:
                shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.output_zip.exists() and self.args.overwrite:
            self.output_zip.unlink()
        self.log(f"Stage 2A Pilot version {VERSION}")
        self.log(f"Input: {self.input_path}")
        self.log(f"Output: {self.output_dir}")

    def discover_tables(self) -> None:
        candidates: list[SourceTable] = []
        if self.input_path.is_file() and zipfile.is_zipfile(self.input_path):
            with zipfile.ZipFile(self.input_path) as z:
                bad = z.testzip()
                if bad:
                    raise zipfile.BadZipFile(f"CRC failure in ZIP member: {bad}")
                names = [i.filename for i in z.infolist() if not i.is_dir() and PurePosixPath(i.filename).suffix.lower() in {".csv", ".tsv", ".txt"}]
            for name in names:
                def opener(member=name, archive=self.input_path):
                    zf = zipfile.ZipFile(archive)
                    raw = zf.open(member, "r")
                    original_close = raw.close
                    def close_both():
                        try:
                            original_close()
                        finally:
                            zf.close()
                    raw.close = close_both  # type: ignore[method-assign]
                    return raw
                headers, delimiter, encoding = read_header(opener)
                hset = set(headers)
                kind = "info" if EXPECTED_INFO.issubset(hset) else "events" if EXPECTED_EVENTS.issubset(hset) else "other"
                candidates.append(SourceTable(name, opener, headers, delimiter, encoding, kind))
        elif self.input_path.is_dir():
            for path in sorted(self.input_path.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
                    continue
                def opener(file_path=path):
                    return file_path.open("rb")
                headers, delimiter, encoding = read_header(opener)
                hset = set(headers)
                kind = "info" if EXPECTED_INFO.issubset(hset) else "events" if EXPECTED_EVENTS.issubset(hset) else "other"
                candidates.append(SourceTable(str(path.relative_to(self.input_path)), opener, headers, delimiter, encoding, kind))
        else:
            raise ValueError("Input must be a ZIP file or a directory")

        info = [c for c in candidates if c.kind == "info"]
        events = [c for c in candidates if c.kind == "events"]
        if len(info) != 1 or len(events) != 1:
            raise ValueError(f"Expected exactly one INFO table and one EVENTS table; found info={len(info)}, events={len(events)}")
        self.info_table, self.event_table = info[0], events[0]
        write_json(self.output_dir / "validated_source_schema.json", {
            "stage1_frozen_schema": True,
            "info": {"source": self.info_table.name, "headers": self.info_table.headers, "delimiter": repr(self.info_table.delimiter), "encoding": self.info_table.encoding},
            "events": {"source": self.event_table.name, "headers": self.event_table.headers, "delimiter": repr(self.event_table.delimiter), "encoding": self.event_table.encoding},
        })
        self.log(f"Validated INFO table: {self.info_table.name}")
        self.log(f"Validated EVENTS table: {self.event_table.name}")

    def open_reader(self, table: SourceTable) -> tuple[BinaryIO, io.TextIOWrapper, csv.DictReader]:
        raw = table.opener()
        text = io.TextIOWrapper(raw, encoding=table.encoding, newline="")
        reader = csv.DictReader(text, delimiter=table.delimiter)
        return raw, text, reader

    def init_db(self) -> None:
        self.conn = sqlite3.connect(self.work_db)
        c = self.conn.cursor()
        c.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE info_raw (
          source_row INTEGER PRIMARY KEY, raw_id TEXT, user TEXT, session_id TEXT,
          date_text TEXT, date_epoch REAL, ua TEXT, screen TEXT, screenh TEXT,
          windoww TEXT, windowh TEXT, documentw TEXT, documenth TEXT
        );
        CREATE TABLE events_raw (
          source_row INTEGER PRIMARY KEY, raw_id TEXT, cursor TEXT, timestamp_text TEXT,
          ts_epoch REAL, xpos REAL, ypos REAL, key_code TEXT, event TEXT, user TEXT, session_id TEXT
        );
        """)
        self.conn.commit()

    def import_sources(self) -> dict:
        assert self.conn and self.info_table and self.event_table
        info_count = event_count = invalid_ts = invalid_xy = 0
        c = self.conn.cursor()
        raw, text, reader = self.open_reader(self.info_table)
        try:
            batch = []
            for i, row in enumerate(reader, start=1):
                batch.append((i, row.get("_id", ""), row.get("user", "").strip(), row.get("session_id", "").strip(),
                              row.get("date", ""), parse_dt(row.get("date", "")), row.get("ua", ""),
                              row.get("screen", ""), row.get("screenh", ""), row.get("windoww", ""),
                              row.get("windowh", ""), row.get("documentw", ""), row.get("documenth", "")))
                if len(batch) >= self.args.batch_size:
                    c.executemany("INSERT INTO info_raw VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                    info_count += len(batch); batch.clear()
            if batch:
                c.executemany("INSERT INTO info_raw VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", batch); info_count += len(batch)
        finally:
            text.close()

        raw, text, reader = self.open_reader(self.event_table)
        try:
            batch = []
            for i, row in enumerate(reader, start=1):
                ts = parse_dt(row.get("timestamp", ""))
                if ts is None:
                    invalid_ts += 1
                try:
                    x = float(row.get("xpos", "")); y = float(row.get("ypos", ""))
                except ValueError:
                    x = y = float("nan"); invalid_xy += 1
                batch.append((i, row.get("_id", ""), row.get("cursor", ""), row.get("timestamp", ""), ts,
                              x, y, row.get("key", ""), row.get("event", "").strip().lower(),
                              row.get("user", "").strip(), row.get("session_id", "").strip()))
                if len(batch) >= self.args.batch_size:
                    c.executemany("INSERT INTO events_raw VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch)
                    event_count += len(batch); batch.clear()
            if batch:
                c.executemany("INSERT INTO events_raw VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch); event_count += len(batch)
        finally:
            text.close()
        self.conn.commit()
        c.executescript("""
        CREATE INDEX idx_info_session ON info_raw(session_id);
        CREATE INDEX idx_events_session_source ON events_raw(session_id, source_row);
        CREATE INDEX idx_events_session_time ON events_raw(session_id, ts_epoch, source_row);
        CREATE INDEX idx_events_user_session ON events_raw(user, session_id);
        """)
        self.conn.commit()
        if invalid_ts:
            self.warn("INVALID_TIMESTAMPS", "critical", f"{invalid_ts} event timestamps could not be parsed")
        if invalid_xy:
            self.warn("INVALID_COORDINATES", "high", f"{invalid_xy} event rows had invalid mouse coordinates")
        self.log(f"Stream-imported {info_count} INFO rows and {event_count} event rows into temporary SQLite")
        return {"info_rows": info_count, "event_rows": event_count, "invalid_event_timestamps": invalid_ts, "invalid_coordinates": invalid_xy}

    def create_clean_events(self) -> dict:
        assert self.conn
        c = self.conn.cursor()
        duplicate_groups = []
        for row in c.execute("""
          SELECT timestamp_text, xpos, ypos, key_code, event, user, session_id,
                 COUNT(*) AS n, MIN(source_row) AS kept_source_row
          FROM events_raw
          GROUP BY timestamp_text, xpos, ypos, key_code, event, user, session_id
          HAVING COUNT(*) > 1
          ORDER BY n DESC, session_id, timestamp_text
        """):
            duplicate_groups.append({
                "session_id": row[6], "user": row[5], "timestamp": row[0], "xpos": row[1], "ypos": row[2],
                "key": row[3], "event": row[4], "duplicate_group_size": row[7],
                "duplicates_removed": row[7] - 1, "kept_source_row": row[8],
            })
        write_csv(self.output_dir / "semantic_duplicate_groups.csv", duplicate_groups,
                  ["session_id","user","timestamp","xpos","ypos","key","event","duplicate_group_size","duplicates_removed","kept_source_row"])
        c.executescript("""
        CREATE TABLE events_clean AS
          SELECT MIN(source_row) AS source_row, MIN(raw_id) AS raw_id, MIN(cursor) AS cursor,
                 timestamp_text, MIN(ts_epoch) AS ts_epoch, xpos, ypos, key_code, event, user, session_id
          FROM events_raw
          GROUP BY timestamp_text, xpos, ypos, key_code, event, user, session_id;
        CREATE INDEX idx_clean_session_time ON events_clean(session_id, ts_epoch, source_row);
        CREATE INDEX idx_clean_user_session ON events_clean(user, session_id);
        """)
        self.conn.commit()
        raw_n = c.execute("SELECT COUNT(*) FROM events_raw").fetchone()[0]
        clean_n = c.execute("SELECT COUNT(*) FROM events_clean").fetchone()[0]
        removed = raw_n - clean_n
        if removed:
            self.warn("SEMANTIC_DUPLICATES_REMOVED", "medium", f"{removed} duplicate event rows are excluded only in derived Stage-2A views")
        return {"raw_event_rows": raw_n, "clean_event_rows": clean_n, "semantic_duplicates_removed": removed, "duplicate_groups": len(duplicate_groups)}

    def build_session_manifest(self) -> dict:
        """Build the complete session catalogue and a diagnostic all-session split.

        The binding primary split is frozen later, after session-level modality
        viability is measured using prespecified thresholds.
        """
        assert self.conn
        c = self.conn.cursor()
        info_by_session: dict[str, dict] = {}
        info_duplicate_ids = []
        for row in c.execute("SELECT session_id, COUNT(*), MIN(user), MAX(user), MIN(date_epoch), MIN(date_text) FROM info_raw GROUP BY session_id"):
            sid, n, min_user, max_user, date_epoch, date_text = row
            info_by_session[sid] = {"info_rows": n, "info_user": min_user, "info_user_max": max_user, "info_date_epoch": date_epoch, "info_date": date_text}
            if n > 1 or min_user != max_user:
                info_duplicate_ids.append({"session_id": sid, "info_rows": n, "min_user": min_user, "max_user": max_user})
        write_csv(self.output_dir / "info_session_conflicts.csv", info_duplicate_ids,
                  ["session_id","info_rows","min_user","max_user"])

        event_rows = []
        for row in c.execute("""
          SELECT session_id, MIN(user), MAX(user), COUNT(*), MIN(ts_epoch), MAX(ts_epoch),
                 SUM(CASE WHEN event IN ('keydown','keyup') THEN 1 ELSE 0 END),
                 SUM(CASE WHEN event NOT IN ('keydown','keyup') THEN 1 ELSE 0 END),
                 COUNT(DISTINCT user)
          FROM events_clean GROUP BY session_id
        """):
            sid, min_user, max_user, n, first_ts, last_ts, key_n, mouse_n, user_n = row
            info = info_by_session.get(sid, {})
            user = min_user
            lower = user.lower()
            enrolled = bool(ENROLLED_USER_RE.match(user))
            preauth = lower == PREAUTH_NAME
            if key_n and mouse_n:
                modality = "keyboard+mouse"
            elif key_n:
                modality = "keyboard_only"
            elif mouse_n:
                modality = "mouse_only"
            else:
                modality = "none"
            event_rows.append({
                "user": user, "session_id": sid, "is_enrolled": int(enrolled), "is_preauth": int(preauth),
                "split": "PENDING" if enrolled else "EXCLUDED_PREAUTH" if preauth else "EXCLUDED_UNKNOWN_USER",
                "candidate_split": "", "candidate_split_order": "", "candidate_split_rule": "",
                "split_order": "", "split_rule": "", "info_present": int(bool(info)),
                "info_user": info.get("info_user", ""), "info_date": info.get("info_date", ""),
                "first_event": fmt_epoch(first_ts), "last_event": fmt_epoch(last_ts),
                "duration_seconds": round(max(0.0, (last_ts or 0) - (first_ts or 0)), 3),
                "event_count_clean": n, "keyboard_event_count": key_n, "mouse_event_count": mouse_n,
                "modality_profile": modality, "distinct_event_users": user_n,
                "user_consistent_with_info": int(bool(info) and info.get("info_user") == user),
                "first_event_epoch": first_ts, "info_date_epoch": info.get("info_date_epoch"),
                "valid_key_pairs": 0, "valid_mouse_strokes": 0, "core_multimodal_eligible": 0,
                "primary_role": "pending" if enrolled else "excluded",
            })
            if user_n != 1 or min_user != max_user:
                self.warn("SESSION_MULTIPLE_USERS", "critical", f"Session {sid} contains multiple event users", sid, user)
            if info and info.get("info_user") != user:
                self.warn("INFO_EVENT_USER_MISMATCH", "critical", f"INFO user={info.get('info_user')} but events user={user}", sid, user)

        grouped: dict[str, list[dict]] = defaultdict(list)
        for r in event_rows:
            if r["is_enrolled"]:
                grouped[r["user"]].append(r)
        candidate_summary = []
        for user in sorted(grouped, key=user_sort_key):
            sessions = sorted(grouped[user], key=lambda r: ((r["first_event_epoch"] if r["first_event_epoch"] is not None else float("inf")), r["session_id"]))
            n_train, n_val, n_test = split_counts(len(sessions))
            for idx, r in enumerate(sessions, start=1):
                if idx <= n_train:
                    split = "train"
                elif idx <= n_train + n_val:
                    split = "validation"
                else:
                    split = "test"
                r["candidate_split"] = split
                r["candidate_split_order"] = idx
                r["candidate_split_rule"] = "diagnostic_all_sessions_chronological_60_20_20"
            candidate_summary.append({
                "user": user, "event_sessions": len(sessions), "candidate_train_sessions": n_train,
                "candidate_validation_sessions": n_val, "candidate_test_sessions": n_test,
            })

        event_sids = {r["session_id"] for r in event_rows}
        excluded = []
        for sid, info in info_by_session.items():
            if sid not in event_sids:
                excluded.append({
                    "user": info.get("info_user", ""), "session_id": sid, "reason": "INFO_ONLY_NO_EVENTS",
                    "info_date": info.get("info_date", ""), "event_count": 0,
                })
        for r in event_rows:
            if not r["is_enrolled"]:
                excluded.append({
                    "user": r["user"], "session_id": r["session_id"],
                    "reason": "PREAUTH_CONTEXT_NOT_ENROLLED_USER" if r["is_preauth"] else "UNKNOWN_USER_NOT_ENROLLED",
                    "info_date": r["info_date"], "event_count": r["event_count_clean"],
                })
        write_csv(self.output_dir / "excluded_sessions.csv", excluded,
                  ["user","session_id","reason","info_date","event_count"])

        candidate_fields = [
            "user","session_id","is_enrolled","is_preauth","candidate_split","candidate_split_order","candidate_split_rule",
            "info_present","info_user","info_date","first_event","last_event","duration_seconds","event_count_clean",
            "keyboard_event_count","mouse_event_count","modality_profile","distinct_event_users","user_consistent_with_info"
        ]
        candidate_rows = [{k: r[k] for k in candidate_fields} for r in sorted(event_rows, key=lambda x: (user_sort_key(x["user"]), x["candidate_split_order"] or 10**9, x["session_id"]))]
        write_csv(self.output_dir / "all_session_chronological_candidate.csv", candidate_rows, candidate_fields)
        write_csv(self.output_dir / "all_session_candidate_split_summary.csv", candidate_summary,
                  ["user","event_sessions","candidate_train_sessions","candidate_validation_sessions","candidate_test_sessions"])
        self.session_rows = event_rows
        self.log(f"Built complete catalogue for {len(event_rows)} event sessions; binding split deferred until modality viability audit")
        return {
            "event_sessions": len(event_rows), "enrolled_users": len(grouped),
            "enrolled_event_sessions": sum(r["is_enrolled"] for r in event_rows),
            "preauth_sessions_excluded": sum(r["is_preauth"] for r in event_rows),
            "info_only_sessions_excluded": sum(1 for x in excluded if x["reason"] == "INFO_ONLY_NO_EVENTS"),
        }

    def get_events(self, session_id: str, clean: bool = True, source_order: bool = False) -> list[dict]:
        assert self.conn
        table = "events_clean" if clean else "events_raw"
        order = "source_row" if source_order else "ts_epoch, source_row"
        rows = []
        for r in self.conn.execute(
            f"SELECT source_row, timestamp_text, ts_epoch, xpos, ypos, key_code, event, user, session_id FROM {table} WHERE session_id=? ORDER BY {order}",
            (session_id,),
        ):
            rows.append({"source_row": r[0], "timestamp": r[1], "ts": r[2], "x": r[3], "y": r[4], "key": r[5], "event": r[6], "user": r[7], "session_id": r[8]})
        return rows

    def temporal_audit(self) -> dict:
        assert self.conn
        rows_out = []
        total_inversions = total_reordered = 0
        duplicate_timestamps_by_session = {r[0]: (r[1], r[2]) for r in self.conn.execute("""
          SELECT session_id, COUNT(*) AS timestamp_groups, SUM(n-1) AS extra_events
          FROM (SELECT session_id, timestamp_text, COUNT(*) AS n FROM events_clean GROUP BY session_id, timestamp_text HAVING COUNT(*) > 1)
          GROUP BY session_id
        """)}
        for s in self.session_rows:
            sid = s["session_id"]
            raw = self.get_events(sid, clean=False, source_order=True)
            valid = [e for e in raw if e["ts"] is not None]
            inversions = sum(1 for a, b in zip(valid, valid[1:]) if b["ts"] < a["ts"])
            source_ids = [e["source_row"] for e in valid]
            sorted_ids = [e["source_row"] for e in sorted(valid, key=lambda e: (e["ts"], e["source_row"]))]
            reordered_positions = sum(a != b for a, b in zip(source_ids, sorted_ids))
            tg, extra = duplicate_timestamps_by_session.get(sid, (0, 0))
            rows_out.append({
                "user": s["user"], "session_id": sid, "split": s["split"],
                "raw_rows": len(raw), "adjacent_timestamp_inversions": inversions,
                "positions_changed_by_stable_sort": reordered_positions,
                "shared_timestamp_groups": tg, "extra_events_at_shared_timestamps": extra,
                "repair_rule": "stable_sort(timestamp,source_row); preserve distinct simultaneous events",
            })
            total_inversions += inversions; total_reordered += reordered_positions
            if inversions:
                self.warn("TEMPORAL_INVERSIONS", "medium", f"{inversions} adjacent inversions repaired by derived stable sorting", sid, s["user"])
        write_csv(self.output_dir / "temporal_repair_audit.csv", rows_out,
                  ["user","session_id","split","raw_rows","adjacent_timestamp_inversions","positions_changed_by_stable_sort","shared_timestamp_groups","extra_events_at_shared_timestamps","repair_rule"])
        return {"sessions_with_inversions": sum(r["adjacent_timestamp_inversions"] > 0 for r in rows_out), "adjacent_inversions": total_inversions, "positions_changed_by_sort": total_reordered}

    def segment_events(self, events: list[dict], threshold: float) -> list[list[dict]]:
        if not events:
            return []
        segments: list[list[dict]] = [[events[0]]]
        for prev, cur in zip(events, events[1:]):
            gap = (cur["ts"] - prev["ts"]) if cur["ts"] is not None and prev["ts"] is not None else float("inf")
            if gap > threshold:
                segments.append([cur])
            else:
                segments[-1].append(cur)
        return segments

    def idle_segmentation(self) -> dict:
        summary_rows = []
        primary_rows = []
        total_by_threshold = Counter()
        for s in self.session_rows:
            events = self.get_events(s["session_id"], clean=True, source_order=False)
            for threshold in self.idle_thresholds:
                segments = self.segment_events(events, threshold)
                total_by_threshold[threshold] += len(segments)
                durations = [(seg[-1]["ts"] - seg[0]["ts"]) if len(seg) > 1 else 0.0 for seg in segments]
                summary_rows.append({
                    "user": s["user"], "session_id": s["session_id"], "split": s["split"],
                    "idle_threshold_seconds": threshold, "segment_count": len(segments),
                    "min_segment_events": min((len(x) for x in segments), default=0),
                    "median_segment_events": round(statistics.median([len(x) for x in segments]), 3) if segments else 0,
                    "max_segment_events": max((len(x) for x in segments), default=0),
                    "max_segment_duration_seconds": round(max(durations, default=0.0), 3),
                })
                if threshold == self.primary_idle:
                    for idx, seg in enumerate(segments, start=1):
                        key_n = sum(e["event"] in KEY_EVENTS for e in seg)
                        mouse_n = len(seg) - key_n
                        seg_id = f"{s['session_id']}_seg{idx:04d}"
                        pr = {
                            "user": s["user"], "session_id": s["session_id"], "segment_id": seg_id,
                            "split": s["split"], "is_enrolled": s["is_enrolled"], "is_preauth": s["is_preauth"],
                            "segment_order": idx, "start_time": fmt_epoch(seg[0]["ts"]), "end_time": fmt_epoch(seg[-1]["ts"]),
                            "duration_seconds": round(max(0.0, seg[-1]["ts"] - seg[0]["ts"]), 3),
                            "event_count": len(seg), "keyboard_event_count": key_n, "mouse_event_count": mouse_n,
                            "modality_profile": "keyboard+mouse" if key_n and mouse_n else "keyboard_only" if key_n else "mouse_only" if mouse_n else "none",
                        }
                        primary_rows.append(pr)
                        self.segment_event_cache[(s["session_id"], idx)] = seg
        write_csv(self.output_dir / "idle_segmentation_sensitivity.csv", summary_rows,
                  ["user","session_id","split","idle_threshold_seconds","segment_count","min_segment_events","median_segment_events","max_segment_events","max_segment_duration_seconds"])
        write_csv(self.output_dir / "primary_segment_manifest_60s.csv", primary_rows,
                  ["user","session_id","segment_id","split","is_enrolled","is_preauth","segment_order","start_time","end_time","duration_seconds","event_count","keyboard_event_count","mouse_event_count","modality_profile"])
        self.primary_segments = primary_rows
        self.log(f"Created {len(primary_rows)} derived segments at primary idle threshold {self.primary_idle:g}s")
        return {"primary_idle_seconds": self.primary_idle, "primary_segments": len(primary_rows), "segment_counts_by_threshold": {str(k): v for k, v in sorted(total_by_threshold.items())}}

    def pair_keyboard(self, events: list[dict]) -> dict:
        pending: dict[str, deque] = defaultdict(deque)
        pair_times: list[float] = []
        dwell: list[float] = []
        valid_dwell: list[float] = []
        keydown_n = keyup_n = orphan_up = 0
        for e in events:
            if e["event"] == "keydown":
                keydown_n += 1
                pending[e["key"]].append(e["ts"])
            elif e["event"] == "keyup":
                keyup_n += 1
                if pending[e["key"]]:
                    down_ts = pending[e["key"]].popleft()
                    d = e["ts"] - down_ts
                    if d >= 0:
                        dwell.append(d)
                        if d <= self.args.max_dwell_seconds:
                            valid_dwell.append(d)
                            pair_times.append(down_ts)
                    else:
                        orphan_up += 1
                else:
                    orphan_up += 1
        orphan_down = sum(len(q) for q in pending.values())
        return {
            "keydown": keydown_n, "keyup": keyup_n, "matched_nonnegative": len(dwell),
            "valid_pairs": len(valid_dwell), "overlong_pairs": len(dwell) - len(valid_dwell),
            "orphan_keydown": orphan_down, "orphan_keyup": orphan_up,
            "dwell_p50_ms": round((percentile(valid_dwell, .50) or 0) * 1000, 3) if valid_dwell else "",
            "dwell_p95_ms": round((percentile(valid_dwell, .95) or 0) * 1000, 3) if valid_dwell else "",
            "dwell_max_ms": round(max(valid_dwell) * 1000, 3) if valid_dwell else "",
            "pair_times": sorted(pair_times),
        }

    def mouse_strokes(self, events: list[dict]) -> dict:
        mouse = [e for e in events if e["event"] in MOUSE_EVENTS]
        dt_values: list[float] = []
        velocities: list[float] = []
        stroke_times: list[float] = []
        zero_dt = invalid_coord = 0
        transitions = Counter()
        for a, b in zip(mouse, mouse[1:]):
            transitions[f"{a['event']}->{b['event']}"] += 1
            dt = b["ts"] - a["ts"]
            if dt <= 0:
                zero_dt += 1
                continue
            if not all(math.isfinite(v) for v in (a["x"], a["y"], b["x"], b["y"])):
                invalid_coord += 1
                continue
            dist = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
            dt_values.append(dt)
            velocities.append(dist / dt)
            stroke_times.append(b["ts"])
        return {
            "mouse_events": len(mouse), "candidate_consecutive_pairs": max(0, len(mouse) - 1),
            "valid_strokes": len(stroke_times), "nonpositive_dt": zero_dt, "invalid_coordinate_pairs": invalid_coord,
            "dt_p50_ms": round((percentile(dt_values, .50) or 0) * 1000, 3) if dt_values else "",
            "dt_p95_ms": round((percentile(dt_values, .95) or 0) * 1000, 3) if dt_values else "",
            "velocity_p50_px_s": round(percentile(velocities, .50) or 0, 3) if velocities else "",
            "top_transition": transitions.most_common(1)[0][0] if transitions else "",
            "stroke_times": sorted(stroke_times),
        }

    def modality_audit(self) -> tuple[dict[tuple[str,int], dict], dict[tuple[str,int], dict], dict]:
        key_rows = []
        mouse_rows = []
        key_cache: dict[tuple[str,int], dict] = {}
        mouse_cache: dict[tuple[str,int], dict] = {}
        totals = Counter()
        all_dwell_pools = []
        for seg in self.primary_segments:
            key = (seg["session_id"], seg["segment_order"])
            events = self.segment_event_cache[key]
            kp = self.pair_keyboard(events)
            ms = self.mouse_strokes(events)
            key_cache[key] = kp; mouse_cache[key] = ms
            key_rows.append({
                "user": seg["user"], "session_id": seg["session_id"], "segment_id": seg["segment_id"], "split": seg["split"],
                "keydown": kp["keydown"], "keyup": kp["keyup"], "matched_nonnegative": kp["matched_nonnegative"],
                "valid_pairs_le_max_dwell": kp["valid_pairs"], "overlong_pairs": kp["overlong_pairs"],
                "orphan_keydown": kp["orphan_keydown"], "orphan_keyup": kp["orphan_keyup"],
                "dwell_p50_ms": kp["dwell_p50_ms"], "dwell_p95_ms": kp["dwell_p95_ms"], "dwell_max_ms": kp["dwell_max_ms"],
                "max_dwell_seconds": self.args.max_dwell_seconds,
            })
            mouse_rows.append({
                "user": seg["user"], "session_id": seg["session_id"], "segment_id": seg["segment_id"], "split": seg["split"],
                "mouse_events": ms["mouse_events"], "candidate_consecutive_pairs": ms["candidate_consecutive_pairs"],
                "valid_strokes": ms["valid_strokes"], "nonpositive_dt": ms["nonpositive_dt"],
                "invalid_coordinate_pairs": ms["invalid_coordinate_pairs"], "dt_p50_ms": ms["dt_p50_ms"],
                "dt_p95_ms": ms["dt_p95_ms"], "velocity_p50_px_s": ms["velocity_p50_px_s"],
                "top_event_transition": ms["top_transition"],
            })
            totals.update({
                "keydown": kp["keydown"], "keyup": kp["keyup"], "valid_key_pairs": kp["valid_pairs"],
                "orphan_keydown": kp["orphan_keydown"], "orphan_keyup": kp["orphan_keyup"],
                "mouse_events": ms["mouse_events"], "valid_mouse_strokes": ms["valid_strokes"], "mouse_nonpositive_dt": ms["nonpositive_dt"],
            })
        write_csv(self.output_dir / "keyboard_pairing_audit.csv", key_rows,
                  ["user","session_id","segment_id","split","keydown","keyup","matched_nonnegative","valid_pairs_le_max_dwell","overlong_pairs","orphan_keydown","orphan_keyup","dwell_p50_ms","dwell_p95_ms","dwell_max_ms","max_dwell_seconds"])
        write_csv(self.output_dir / "mouse_stroke_audit.csv", mouse_rows,
                  ["user","session_id","segment_id","split","mouse_events","candidate_consecutive_pairs","valid_strokes","nonpositive_dt","invalid_coordinate_pairs","dt_p50_ms","dt_p95_ms","velocity_p50_px_s","top_event_transition"])
        if totals["mouse_nonpositive_dt"]:
            self.warn("SIMULTANEOUS_MOUSE_EVENTS", "medium", f"{totals['mouse_nonpositive_dt']} consecutive mouse pairs have non-positive dt and are excluded from derived strokes")
        return key_cache, mouse_cache, dict(totals)


    def freeze_core_manifest(self, key_cache: dict, mouse_cache: dict, base_stats: dict) -> dict:
        """Freeze the primary split on prespecified core multimodal sessions.

        A core session must have at least the configured number of valid
        keyboard pairs and mouse strokes after temporal repair and primary idle
        segmentation. This is an inclusion/quality rule, not a label- or
        performance-based selection rule.
        """
        key_totals = Counter()
        mouse_totals = Counter()
        for (sid, seg_order), stats in key_cache.items():
            key_totals[sid] += int(stats["valid_pairs"])
        for (sid, seg_order), stats in mouse_cache.items():
            mouse_totals[sid] += int(stats["valid_strokes"])

        core_by_user: dict[str, list[dict]] = defaultdict(list)
        auxiliary = []
        for r in self.session_rows:
            r["valid_key_pairs"] = key_totals[r["session_id"]]
            r["valid_mouse_strokes"] = mouse_totals[r["session_id"]]
            eligible = bool(
                r["is_enrolled"]
                and r["valid_key_pairs"] >= self.args.core_min_key_pairs
                and r["valid_mouse_strokes"] >= self.args.core_min_mouse_strokes
            )
            r["core_multimodal_eligible"] = int(eligible)
            if eligible:
                r["primary_role"] = "core_primary"
                core_by_user[r["user"]].append(r)
            elif r["is_enrolled"]:
                r["primary_role"] = "auxiliary_only"
                r["split"] = "auxiliary_only"
                reasons = []
                if r["valid_key_pairs"] < self.args.core_min_key_pairs:
                    reasons.append("INSUFFICIENT_VALID_KEY_PAIRS")
                if r["valid_mouse_strokes"] < self.args.core_min_mouse_strokes:
                    reasons.append("INSUFFICIENT_VALID_MOUSE_STROKES")
                auxiliary.append({
                    "user": r["user"], "session_id": r["session_id"], "candidate_split": r["candidate_split"],
                    "valid_key_pairs": r["valid_key_pairs"], "valid_mouse_strokes": r["valid_mouse_strokes"],
                    "reason": "+".join(reasons), "retained_for": "modality-specific/robustness analysis only",
                })

        user_summary = []
        for user in sorted(core_by_user, key=user_sort_key):
            sessions = sorted(core_by_user[user], key=lambda r: ((r["first_event_epoch"] if r["first_event_epoch"] is not None else float("inf")), r["session_id"]))
            n_train, n_val, n_test = split_counts(len(sessions))
            if n_val == 0 or n_test == 0:
                self.warn("INSUFFICIENT_CORE_SESSIONS", "critical", f"{user} has only {len(sessions)} core multimodal sessions", user=user)
            for idx, r in enumerate(sessions, start=1):
                if idx <= n_train:
                    split = "train"
                elif idx <= n_train + n_val:
                    split = "validation"
                else:
                    split = "test"
                r["split"] = split
                r["split_order"] = idx
                r["split_rule"] = "core_multimodal_chronological_60_20_20_min_one_val_test"
            total_event = sum(1 for r in self.session_rows if r["user"] == user and r["is_enrolled"])
            user_summary.append({
                "user": user, "event_sessions": total_event, "core_multimodal_sessions": len(sessions),
                "auxiliary_sessions": total_event - len(sessions), "train_sessions": n_train,
                "validation_sessions": n_val, "test_sessions": n_test,
                "adequate_core_session_subset": int(len(sessions) >= 5),
            })
            if len(sessions) < 5:
                self.warn("SPARSE_CORE_USER_SESSIONS", "high", f"{user} has {len(sessions)} core multimodal sessions; include but flag sensitivity analysis", user=user)

        missing_users = [f"User{i}" for i in range(1,12) if f"User{i}" not in core_by_user]
        if missing_users:
            self.warn("USERS_WITHOUT_CORE_SESSIONS", "critical", f"No core sessions for: {', '.join(missing_users)}")

        split_by_sid = {r["session_id"]: r["split"] for r in self.session_rows}
        for seg in self.primary_segments:
            seg["split"] = split_by_sid.get(seg["session_id"], seg["split"])

        public_fields = [
            "user","session_id","is_enrolled","is_preauth","primary_role","core_multimodal_eligible",
            "valid_key_pairs","valid_mouse_strokes","split","split_order","split_rule",
            "candidate_split","candidate_split_order","info_present","info_user","info_date","first_event","last_event",
            "duration_seconds","event_count_clean","keyboard_event_count","mouse_event_count","modality_profile",
            "distinct_event_users","user_consistent_with_info"
        ]
        public_rows = [{k: r[k] for k in public_fields} for r in sorted(self.session_rows, key=lambda x: (user_sort_key(x["user"]), x["split_order"] or 10**9, x["session_id"]))]
        write_csv(self.output_dir / "frozen_session_manifest.csv", public_rows, public_fields)
        write_csv(self.output_dir / "user_split_summary.csv", user_summary,
                  ["user","event_sessions","core_multimodal_sessions","auxiliary_sessions","train_sessions","validation_sessions","test_sessions","adequate_core_session_subset"])
        write_csv(self.output_dir / "auxiliary_sessions.csv", auxiliary,
                  ["user","session_id","candidate_split","valid_key_pairs","valid_mouse_strokes","reason","retained_for"])
        write_csv(self.output_dir / "primary_segment_manifest_60s.csv", self.primary_segments,
                  ["user","session_id","segment_id","split","is_enrolled","is_preauth","segment_order","start_time","end_time","duration_seconds","event_count","keyboard_event_count","mouse_event_count","modality_profile"])

        # Update split annotations in already-generated audit tables.
        for filename in ["temporal_repair_audit.csv", "idle_segmentation_sensitivity.csv", "keyboard_pairing_audit.csv", "mouse_stroke_audit.csv"]:
            path = self.output_dir / filename
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f)); fields = list(rows[0].keys()) if rows else []
            for row in rows:
                row["split"] = split_by_sid.get(row.get("session_id", ""), row.get("split", ""))
            write_csv(path, rows, fields)

        core_total = sum(r["core_multimodal_eligible"] for r in self.session_rows)
        self.log(f"Frozen binding core split for {len(core_by_user)} users and {core_total} core multimodal sessions")
        return {
            **base_stats,
            "core_min_key_pairs": self.args.core_min_key_pairs,
            "core_min_mouse_strokes": self.args.core_min_mouse_strokes,
            "core_multimodal_sessions": core_total,
            "auxiliary_enrolled_sessions": len(auxiliary),
            "adequate_core_session_users": [r["user"] for r in user_summary if r["adequate_core_session_subset"]],
            "sparse_core_session_users": [r["user"] for r in user_summary if not r["adequate_core_session_subset"]],
        }

    def time_window_counts(self, start: float, end: float, pair_times: list[float], stroke_times: list[float], duration: float, min_key: int, min_mouse: int) -> dict:
        step = duration / 2.0
        full = both = any_mod = key_only = mouse_only = 0
        if end - start + 1e-9 < duration:
            return {"candidate_windows": 0, "eligible_any": 0, "eligible_both": 0, "keyboard_only_eligible": 0, "mouse_only_eligible": 0}
        wstart = start
        while wstart + duration <= end + 1e-9:
            wend = wstart + duration
            full += 1
            k = bisect.bisect_left(pair_times, wend) - bisect.bisect_left(pair_times, wstart)
            m = bisect.bisect_left(stroke_times, wend) - bisect.bisect_left(stroke_times, wstart)
            key_ok, mouse_ok = k >= min_key, m >= min_mouse
            if key_ok or mouse_ok:
                any_mod += 1
            if key_ok and mouse_ok:
                both += 1
            elif key_ok:
                key_only += 1
            elif mouse_ok:
                mouse_only += 1
            wstart += step
        return {"candidate_windows": full, "eligible_any": any_mod, "eligible_both": both, "keyboard_only_eligible": key_only, "mouse_only_eligible": mouse_only}

    def window_feasibility(self, key_cache: dict, mouse_cache: dict) -> dict:
        session_acc: dict[tuple[str,str,str], Counter] = defaultdict(Counter)
        config_meta: dict[str, dict] = {}
        fixed_sizes = [int(x) for x in self.args.fixed_sizes.split(",") if x.strip()]
        for size in fixed_sizes:
            step = max(1, size // 2)
            config_meta[f"keyboard_fixed_{size}_pairs_50pct"] = {"family":"keyboard_fixed", "size":size, "step":step, "primary_metric":"candidate_windows"}
            config_meta[f"mouse_fixed_{size}_strokes_50pct"] = {"family":"mouse_fixed", "size":size, "step":step, "primary_metric":"candidate_windows"}
        time_cfgs = [(10.0,3,3), (30.0,5,10), (60.0,10,20)]
        for duration, mk, mm in time_cfgs:
            config_meta[f"time_{int(duration)}s_50pct_minK{mk}_minM{mm}"] = {"family":"time_multimodal", "duration":duration, "min_key_pairs":mk, "min_mouse_strokes":mm, "primary_metric":"eligible_both"}

        for seg in self.primary_segments:
            if seg["split"] not in {"train", "validation", "test"}:
                continue
            key = (seg["session_id"], seg["segment_order"])
            kp = key_cache[key]; ms = mouse_cache[key]
            base = (seg["user"], seg["session_id"], seg["split"])
            for size in fixed_sizes:
                name = f"keyboard_fixed_{size}_pairs_50pct"
                session_acc[(base[0],base[1],base[2],name)]["candidate_windows"] += sliding_count(kp["valid_pairs"], size, max(1,size//2))
                name = f"mouse_fixed_{size}_strokes_50pct"
                session_acc[(base[0],base[1],base[2],name)]["candidate_windows"] += sliding_count(ms["valid_strokes"], size, max(1,size//2))
            events = self.segment_event_cache[key]
            start, end = events[0]["ts"], events[-1]["ts"]
            for duration, mk, mm in time_cfgs:
                name = f"time_{int(duration)}s_50pct_minK{mk}_minM{mm}"
                counts = self.time_window_counts(start, end, kp["pair_times"], ms["stroke_times"], duration, mk, mm)
                session_acc[(base[0],base[1],base[2],name)].update(counts)

        session_rows = []
        for (user,sid,split,config), counts in sorted(session_acc.items(), key=lambda x:(user_sort_key(x[0][0]), x[0][2], x[0][1], x[0][3])):
            meta = config_meta[config]
            session_rows.append({
                "user":user,"session_id":sid,"split":split,"config":config,"family":meta["family"],
                "candidate_windows":counts.get("candidate_windows",0),"eligible_any":counts.get("eligible_any",0),
                "eligible_both":counts.get("eligible_both",0),"keyboard_only_eligible":counts.get("keyboard_only_eligible",0),
                "mouse_only_eligible":counts.get("mouse_only_eligible",0),
            })
        write_csv(self.output_dir / "window_feasibility_by_session.csv", session_rows,
                  ["user","session_id","split","config","family","candidate_windows","eligible_any","eligible_both","keyboard_only_eligible","mouse_only_eligible"])
        self.window_session_rows = session_rows

        agg: dict[tuple[str,str,str], Counter] = defaultdict(Counter)
        sessions_with: dict[tuple[str,str,str], set] = defaultdict(set)
        for r in session_rows:
            key = (r["user"],r["split"],r["config"])
            for col in ("candidate_windows","eligible_any","eligible_both","keyboard_only_eligible","mouse_only_eligible"):
                agg[key][col] += int(r[col])
            metric = "eligible_both" if r["family"] == "time_multimodal" else "candidate_windows"
            if int(r[metric]) > 0:
                sessions_with[key].add(r["session_id"])
        user_split_rows = []
        for key, counts in sorted(agg.items(), key=lambda x:(user_sort_key(x[0][0]), x[0][1], x[0][2])):
            user, split, config = key
            family = config_meta[config]["family"]
            metric = config_meta[config]["primary_metric"]
            value = counts[metric]
            row = {
                "user":user,"split":split,"config":config,"family":family,
                "candidate_windows":counts["candidate_windows"],"eligible_any":counts["eligible_any"],
                "eligible_both":counts["eligible_both"],"keyboard_only_eligible":counts["keyboard_only_eligible"],
                "mouse_only_eligible":counts["mouse_only_eligible"],"sessions_with_primary_windows":len(sessions_with[key]),
                "primary_metric":metric,"primary_window_count":value,
            }
            user_split_rows.append(row)
            if value == 0:
                self.coverage_warnings.append({
                    "severity":"high","user":user,"split":split,"config":config,"family":family,
                    "missing_metric":metric,"message":"No usable windows under this candidate configuration",
                })
        write_csv(self.output_dir / "window_feasibility_by_user_split.csv", user_split_rows,
                  ["user","split","config","family","candidate_windows","eligible_any","eligible_both","keyboard_only_eligible","mouse_only_eligible","sessions_with_primary_windows","primary_metric","primary_window_count"])
        write_csv(self.output_dir / "window_coverage_warnings.csv", self.coverage_warnings,
                  ["severity","user","split","config","family","missing_metric","message"])
        self.window_user_split_rows = user_split_rows

        config_summary = []
        for config, meta in config_meta.items():
            subset = [r for r in user_split_rows if r["config"] == config]
            metric = meta["primary_metric"]
            config_summary.append({
                "config":config,"family":meta["family"],"primary_metric":metric,
                "total_primary_windows":sum(int(r[metric]) for r in subset),
                "covered_user_split_cells":sum(int(r[metric]) > 0 for r in subset),
                "total_user_split_cells":len(subset),
                "coverage_rate":round(sum(int(r[metric]) > 0 for r in subset)/max(1,len(subset)),6),
                "zero_cells":sum(int(r[metric]) == 0 for r in subset),
            })
        write_csv(self.output_dir / "window_configuration_summary.csv", config_summary,
                  ["config","family","primary_metric","total_primary_windows","covered_user_split_cells","total_user_split_cells","coverage_rate","zero_cells"])
        self.log(f"Evaluated {len(config_meta)} window configurations without materializing model inputs")
        return {"configurations": config_summary, "coverage_warning_count": len(self.coverage_warnings)}

    def write_reports(self, import_stats: dict, duplicate_stats: dict, manifest_stats: dict, temporal_stats: dict, segment_stats: dict, modality_stats: dict, window_stats: dict) -> None:
        manifest_path = self.output_dir / "frozen_session_manifest.csv"
        split_hash = sha256_file(manifest_path)
        write_json(self.output_dir / "split_protocol.json", {
            "protocol_name":"chronological_session_split_v1",
            "frozen_manifest_sha256":split_hash,
            "eligible_population":"Prespecified core multimodal sessions from User1..User11",
            "excluded_from_enrollment":["Alluserspreauth", "INFO-only sessions", "unknown user labels"],
            "ordering":"first_event_timestamp ascending; session_id tie-break",
            "allocation":"After fixed modality-viability inclusion, chronological 60/20/20 using half-up rounding, with explicit 1/1/1 for n=3 and 2/1/1 for n=4",
            "leakage_guards":[
                "All derived idle segments remain in the split of their original core session",
                "No window may cross an original session or idle-segment boundary",
                "No scaler, encoder, calibration, or threshold fitting is performed in Stage 2A",
                "Raw identifiers and environment/device fields are metadata only and prohibited as model features",
            ],
        })
        summary = {
            "stage":"Keystroke–Mouse Stage 2A",
            "version":VERSION,
            "status":"PASS_WITH_WARNINGS" if self.warnings and not self.failures else "PASS" if not self.failures else "FAIL",
            "source_sha256": sha256_file(self.input_path) if self.input_path.is_file() else "directory_input",
            "frozen_manifest_sha256":split_hash,
            "parameters":{
                "idle_thresholds_seconds":self.idle_thresholds,"primary_idle_seconds":self.primary_idle,
                "max_dwell_seconds":self.args.max_dwell_seconds,"fixed_sizes":self.args.fixed_sizes,
                "core_min_key_pairs":self.args.core_min_key_pairs,"core_min_mouse_strokes":self.args.core_min_mouse_strokes,
                "batch_size":self.args.batch_size,
            },
            "import":import_stats,"duplicate_control":duplicate_stats,"session_manifest":manifest_stats,
            "temporal_repair":temporal_stats,"idle_segmentation":segment_stats,"modality_audit":modality_stats,
            "window_feasibility":window_stats,"warning_count":len(self.warnings),"failure_count":len(self.failures),
            "training_performed":False,"claims_created":False,"source_modified":False,
        }
        write_json(self.output_dir / "stage2a_summary.json", summary)

        config_rows = window_stats["configurations"]
        time_rows = [r for r in config_rows if r["family"] == "time_multimodal"]
        fixed_key = [r for r in config_rows if r["family"] == "keyboard_fixed"]
        fixed_mouse = [r for r in config_rows if r["family"] == "mouse_fixed"]
        def numeric_scale(row: dict) -> int:
            nums = re.findall(r"\d+", row["config"])
            return int(nums[0]) if nums else 0

        # Prefer complete user–split coverage, then the largest context window;
        # only then use total window count as a tie-breaker.
        best_time = max(time_rows, key=lambda r:(r["covered_user_split_cells"], numeric_scale(r), r["total_primary_windows"])) if time_rows else None
        best_key = max(fixed_key, key=lambda r:(r["covered_user_split_cells"], numeric_scale(r), r["total_primary_windows"])) if fixed_key else None
        best_mouse = max(fixed_mouse, key=lambda r:(r["covered_user_split_cells"], numeric_scale(r), r["total_primary_windows"])) if fixed_mouse else None
        rec = [
            "# Stage 2A — Empirical Windowing Recommendation",
            "",
            "## Frozen decisions",
            "",
            "- Use only User1–User11 as enrolled identities.",
            f"- Primary core sessions require at least {self.args.core_min_key_pairs} valid keyboard pairs and {self.args.core_min_mouse_strokes} valid mouse strokes; non-core enrolled sessions remain auxiliary-only.",
            "- Keep `Alluserspreauth` outside the primary authentication experiment.",
            "- Freeze the chronological session split in `frozen_session_manifest.csv`; do not regenerate it after viewing test results.",
            f"- Split each original session at idle gaps greater than {self.primary_idle:g} seconds; every derived segment inherits the original session split.",
            "- Remove only the exact semantic duplicate rows listed in `semantic_duplicate_groups.csv` from derived processing.",
            f"- Accept keydown/keyup pairs only when dwell time is non-negative and no greater than {self.args.max_dwell_seconds:g} seconds.",
            "- Define mouse strokes from consecutive mouse events with positive elapsed time; do not use raw click duration as a core feature in the first experiment.",
            "",
            "## Candidate coverage observed",
            "",
        ]
        for r in config_rows:
            rec.append(f"- `{r['config']}`: {r['total_primary_windows']} primary windows; coverage {r['covered_user_split_cells']}/{r['total_user_split_cells']} user–split cells; zero cells {r['zero_cells']}.")
        rec += ["", "## Recommended Stage 2B inputs", ""]
        if best_time:
            rec.append(f"- Multimodal synchronized candidate: `{best_time['config']}`. Use modality masks and report both `eligible_both` and `eligible_any`; do not silently discard single-modality windows.")
        if best_key:
            rec.append(f"- Keyboard-only baseline candidate: `{best_key['config']}`.")
        if best_mouse:
            rec.append(f"- Mouse-only baseline candidate: `{best_mouse['config']}`.")
        rec += [
            "- Stage 2B must create feature tables from Training/Validation/Test separately and fit all learned preprocessing on Training only.",
            "- Do not create genuine/impostor claims or train models until the Stage 2A review is accepted.",
        ]
        (self.output_dir / "recommended_stage2b_configuration.md").write_text("\n".join(rec), encoding="utf-8")

        leakage = f"""# Stage 2A Leakage-Control Report

## Binding split rule

The session-level split is frozen by the SHA-256 value `{split_hash}`. All later segments, windows, feature rows, claims, scores, calibration records, and thresholds must retain the original `session_id` and its frozen split.

## Prohibited leakage paths

1. A segment or window from one original session may not appear in more than one split.
2. Overlapping windows may be generated only after splitting and only inside one idle segment.
3. `user`, `session_id`, `_id`, raw timestamp, user-agent, screen/window/document dimensions, cursor, file path, and source-row order are metadata and must not enter model features.
4. Duplicate filtering is frozen before window construction; duplicate-derived windows must not be generated.
5. Scalers, encoders, feature selection, imputation, calibration, and personal thresholds must be fitted without test data.
6. Genuine and impostor claims must be constructed independently inside Validation and Test after the window manifest is frozen.
7. `Alluserspreauth` is not a twelfth enrolled user and may not be mixed into genuine claims.
8. Test coverage must not be used to choose a more favorable split or to retune window parameters; parameter alternatives must be reported as a prespecified feasibility/sensitivity analysis.

## Stage status

No model, claim, calibration, or threshold was created in this stage.
"""
        (self.output_dir / "stage2a_leakage_control_report.md").write_text(leakage, encoding="utf-8")

        all_issues = self.warnings + self.failures
        write_csv(self.output_dir / "warnings_and_failures.csv", all_issues,
                  ["type","code","severity","user","session_id","message"])
        (self.output_dir / "execution_log.txt").write_text("\n".join(self.logs) + "\n", encoding="utf-8")

    def finalize(self) -> None:
        if self.conn:
            self.conn.close(); self.conn = None
        if self.work_db.exists() and not self.args.keep_work_db:
            self.work_db.unlink()
        self.log(f"Final review ZIP target: {self.output_zip}")
        (self.output_dir / "execution_log.txt").write_text("\n".join(self.logs) + "\n", encoding="utf-8")
        rows = []
        for f in sorted(self.output_dir.iterdir()):
            if f.is_file() and f.name != "output_manifest_sha256.csv":
                rows.append({"file":f.name,"bytes":f.stat().st_size,"sha256":sha256_file(f)})
        write_csv(self.output_dir / "output_manifest_sha256.csv", rows, ["file","bytes","sha256"])
        with zipfile.ZipFile(self.output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for f in sorted(self.output_dir.iterdir()):
                if f.is_file():
                    z.write(f, arcname=f.name)

    def run(self) -> int:
        try:
            self.prepare_output()
            self.discover_tables()
            self.init_db()
            import_stats = self.import_sources()
            duplicate_stats = self.create_clean_events()
            base_manifest_stats = self.build_session_manifest()
            temporal_stats = self.temporal_audit()
            segment_stats = self.idle_segmentation()
            key_cache, mouse_cache, modality_stats = self.modality_audit()
            manifest_stats = self.freeze_core_manifest(key_cache, mouse_cache, base_manifest_stats)
            window_stats = self.window_feasibility(key_cache, mouse_cache)
            self.write_reports(import_stats, duplicate_stats, manifest_stats, temporal_stats, segment_stats, modality_stats, window_stats)
            self.finalize()
            print(f"Stage 2A completed: {self.output_zip}")
            return 0
        except Exception as exc:
            self.fail("UNHANDLED_EXCEPTION", f"{type(exc).__name__}: {exc}")
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                (self.output_dir / "execution_log.txt").write_text("\n".join(self.logs) + "\n" + traceback.format_exc(), encoding="utf-8")
                write_csv(self.output_dir / "warnings_and_failures.csv", self.warnings + self.failures,
                          ["type","code","severity","user","session_id","message"])
            except Exception:
                pass
            print(f"Stage 2A failed: {exc}", file=sys.stderr)
            return 2
        finally:
            if self.conn:
                self.conn.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Keystroke–Mouse Stage 2A safe pilot")
    p.add_argument("--input", required=True, help="Source UEBA ZIP file or extracted dataset directory")
    p.add_argument("--output", required=True, help="New review output directory")
    p.add_argument("--idle-thresholds", default="30,60,120", help="Comma-separated idle-gap sensitivity thresholds in seconds")
    p.add_argument("--primary-idle-seconds", type=float, default=60.0, help="Primary idle gap used for segment manifest")
    p.add_argument("--max-dwell-seconds", type=float, default=2.0, help="Maximum key dwell retained as a valid pair")
    p.add_argument("--fixed-sizes", default="5,10,25,50", help="Comma-separated fixed key-pair/mouse-stroke window sizes")
    p.add_argument("--core-min-key-pairs", type=int, default=5, help="Minimum valid key pairs for a primary core session")
    p.add_argument("--core-min-mouse-strokes", type=int, default=5, help="Minimum valid mouse strokes for a primary core session")
    p.add_argument("--batch-size", type=int, default=10000, help="SQLite insertion batch size")
    p.add_argument("--keep-work-db", action="store_true", help="Keep temporary SQLite work database in output")
    p.add_argument("--overwrite", action="store_true", help="Replace a pre-existing output directory")
    return p


if __name__ == "__main__":
    raise SystemExit(Pilot(build_parser().parse_args()).run())

from __future__ import annotations

import re
from pathlib import Path

from .common import SessionRecord, read_records, stable_session_id


TRACE_SUFFIXES = {".pcap", ".pcapng", ".npz", ".csv"}
EXCLUDED_CSV = {"qoe.csv", "quality.csv", "application_data.csv", "stats_for_nerds.csv"}


def _companion_qoe(trace: Path) -> str:
    candidates = sorted(
        path
        for path in trace.parent.glob("*.csv")
        if path != trace and re.search(r"qoe|quality|playback|representation|bitrate", path.name, re.I)
    )
    return str(candidates[0]) if candidates else ""


def _resolve_manifest(root: Path, manifest: Path) -> list[SessionRecord]:
    records = read_records(manifest)
    for record in records:
        trace = Path(record.trace_path)
        record.trace_path = str(trace if trace.is_absolute() else (root / trace).resolve())
        if record.qoe_path:
            qoe = Path(record.qoe_path)
            record.qoe_path = str(qoe if qoe.is_absolute() else (root / qoe).resolve())
    return records


def discover_longenough(root: str | Path, manifest_name: str = "sessions.csv") -> list[SessionRecord]:
    """Discover LongEnough traces or load an explicit dataset manifest.

    The preferred manifest columns are documented in the project README. Automatic
    discovery supports the official layout with one directory per video and nested
    variable-bandwidth trace files.
    """
    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"LongEnough root not found: {dataset_root}")
    manifest = dataset_root / manifest_name
    if manifest.is_file():
        return _resolve_manifest(dataset_root, manifest)
    records: list[SessionRecord] = []
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TRACE_SUFFIXES:
            continue
        if path.name.lower() in EXCLUDED_CSV or "qoe" in path.name.lower():
            continue
        relative = path.relative_to(dataset_root)
        parts = relative.parts
        bandwidth = next((match.group(0).lower() for part in parts for match in [re.search(r"bw[1248]", part, re.I)] if match), "unknown")
        non_condition = [part for part in parts[:-1] if not re.fullmatch(r"(?:bw[1248]|offset[-_]?0|undefended|none[-_]none)", part, re.I)]
        if not non_condition:
            raise ValueError(f"Could not infer video identity from {relative}; provide sessions.csv")
        video_id = non_condition[0]
        session_id = stable_session_id(path, dataset_root)
        record = SessionRecord(
            session_id=session_id,
            video_id=str(video_id),
            trace_path=str(path),
            bandwidth=bandwidth,
            duplicate_group=session_id,
            qoe_path=_companion_qoe(path),
        )
        records.append(record)
    if not records:
        raise FileNotFoundError(
            f"No LongEnough packet traces found below {dataset_root}. Provide PCAP/CSV/NPZ traces or sessions.csv."
        )
    return records

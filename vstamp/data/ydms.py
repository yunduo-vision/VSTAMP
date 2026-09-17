from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import SessionRecord, read_records, stable_session_id


VIDEO_ID_ALIASES = ("videoid", "video_id", "video id", "content_id")
STATE_ALIASES = ("phase", "player_state", "buffer_state", "state", "playback_state")


def _abr_mode_from_application(path: Path) -> str:
    if not path.is_file():
        return "unknown"
    frame = pd.read_csv(path)
    lower = {str(column).strip().lower(): str(column) for column in frame.columns}
    state_column = next((lower[alias] for alias in STATE_ALIASES if alias in lower), None)
    if state_column is None:
        return "unknown"
    states = frame[state_column].dropna().astype(str).str.strip().str.lower()
    if states.empty:
        return "unknown"
    starved = states.str.contains("starv|stall|rebuffer|deplet", regex=True).mean()
    stable = states.str.contains("steady|stable|normal", regex=True).mean()
    if starved >= 0.20:
        return "starved"
    if stable >= 0.80:
        return "high_stable"
    return "transitional"


def _video_id_from_application(measurement: Path) -> str:
    for name in ("application_data.csv", "stats_for_nerds.csv"):
        path = measurement / name
        if not path.is_file():
            continue
        frame = pd.read_csv(path, nrows=100)
        lower = {str(column).strip().lower(): str(column) for column in frame.columns}
        column = next((lower[alias] for alias in VIDEO_ID_ALIASES if alias in lower), None)
        if column:
            values = frame[column].dropna().astype(str)
            if not values.empty:
                return values.mode().iloc[0]
    raise ValueError(f"Cannot infer video ID for {measurement}; provide sessions.csv")


def discover_ydms(root: str | Path, manifest_name: str = "sessions.csv") -> list[SessionRecord]:
    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"YDMS root not found: {dataset_root}")
    manifest = dataset_root / manifest_name
    if manifest.is_file():
        records = read_records(manifest)
        for record in records:
            trace = Path(record.trace_path)
            record.trace_path = str(trace if trace.is_absolute() else (dataset_root / trace).resolve())
            if record.qoe_path:
                qoe = Path(record.qoe_path)
                record.qoe_path = str(qoe if qoe.is_absolute() else (dataset_root / qoe).resolve())
                if record.abr_mode == "unknown":
                    record.abr_mode = _abr_mode_from_application(Path(record.qoe_path))
        return records
    records: list[SessionRecord] = []
    for trace in dataset_root.rglob("video_traffic.csv"):
        measurement = trace.parent
        video_id = _video_id_from_application(measurement)
        qoe = measurement / "application_data.csv"
        session_id = stable_session_id(trace, dataset_root)
        records.append(
            SessionRecord(
                session_id=session_id,
                video_id=video_id,
                trace_path=str(trace),
                duplicate_group=session_id,
                qoe_path=str(qoe) if qoe.is_file() else "",
                abr_mode=_abr_mode_from_application(qoe),
                extra={"measurement_dir": str(measurement)},
            )
        )
    if not records:
        raise FileNotFoundError(
            f"No YDMS video_traffic.csv files found below {dataset_root}; provide the official layout or sessions.csv."
        )
    return records

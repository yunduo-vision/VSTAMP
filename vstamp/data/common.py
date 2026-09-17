from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PacketTrace:
    timestamps: np.ndarray
    directions: np.ndarray
    lengths: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        if len(self.directions) != n or len(self.lengths) != n:
            raise ValueError("Packet arrays must have the same length")
        if n and np.any(np.diff(self.timestamps) < 0):
            raise ValueError("Packet timestamps must be sorted")
        if not np.all(np.isin(self.directions, (-1, 1))):
            raise ValueError("Directions must use -1 upstream and +1 downstream")
        if np.any(self.lengths < 0):
            raise ValueError("Packet lengths must be non-negative")

    @property
    def duration(self) -> float:
        return 0.0 if len(self.timestamps) < 2 else float(self.timestamps[-1] - self.timestamps[0])


@dataclass
class SessionRecord:
    session_id: str
    video_id: str
    trace_path: str
    bandwidth: str = "unknown"
    abr_mode: str = "unknown"
    split: str = ""
    duplicate_group: str = ""
    client_ip: str = ""
    qoe_path: str = ""
    feature_path: str = ""
    timestamp_scale: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["extra"] = json.dumps(row["extra"], sort_keys=True)
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SessionRecord":
        values = dict(row)
        raw_extra = values.get("extra", {})
        if isinstance(raw_extra, str):
            values["extra"] = json.loads(raw_extra) if raw_extra.strip() else {}
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: values[key] for key in allowed if key in values})


TIMESTAMP_ALIASES = ("timestamp", "time", "ts", "frame.time_epoch", "relative_timestamp")
DIRECTION_ALIASES = ("direction", "dir", "packet_direction")
LENGTH_ALIASES = (
    "length",
    "packet_length",
    "frame.len",
    "payload_length",
    "payloadLength",
    "udp.length",
    "tcp.len",
)
PAYLOAD_LENGTH_ALIASES = (
    "payload_length",
    "payloadLength",
    "tcp.len",
    "udp.length",
    "length",
    "packet_length",
    "frame.len",
)
SOURCE_ALIASES = ("src_ip", "source", "ip.src", "src")
DESTINATION_ALIASES = ("dst_ip", "destination", "ip.dst", "dst")


def _column(frame: pd.DataFrame, aliases: Iterable[str], purpose: str) -> str:
    lower = {str(name).strip().lower(): str(name) for name in frame.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    raise ValueError(f"Could not find {purpose} column; accepted aliases: {list(aliases)}")


def _parse_direction(values: pd.Series) -> np.ndarray:
    if np.issubdtype(values.dtype, np.number):
        raw = values.to_numpy(dtype=np.float64)
        return np.where(raw > 0, 1, -1).astype(np.int8)
    mapping = {
        "down": 1,
        "downstream": 1,
        "in": 1,
        "incoming": 1,
        "1": 1,
        "up": -1,
        "upstream": -1,
        "out": -1,
        "outgoing": -1,
        "-1": -1,
    }
    parsed = values.astype(str).str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        bad = sorted(values[parsed.isna()].astype(str).unique().tolist())[:5]
        raise ValueError(f"Unrecognized direction values: {bad}")
    return parsed.to_numpy(dtype=np.int8)


def load_csv_trace(
    path: str | Path, client_ip: str = "", prefer_payload_length: bool = False
) -> PacketTrace:
    frame = pd.read_csv(path)
    time_col = _column(frame, TIMESTAMP_ALIASES, "timestamp")
    aliases = PAYLOAD_LENGTH_ALIASES if prefer_payload_length else LENGTH_ALIASES
    length_col = _column(frame, aliases, "packet length")
    try:
        direction_col = _column(frame, DIRECTION_ALIASES, "direction")
        directions = _parse_direction(frame[direction_col])
    except ValueError:
        if not client_ip:
            raise ValueError(
                f"{path} has no direction column; provide client_ip in sessions.csv to infer direction"
            )
        src_col = _column(frame, SOURCE_ALIASES, "source IP")
        dst_col = _column(frame, DESTINATION_ALIASES, "destination IP")
        src = frame[src_col].astype(str).str.strip()
        dst = frame[dst_col].astype(str).str.strip()
        valid = (src == client_ip) | (dst == client_ip)
        frame = frame.loc[valid].copy()
        directions = np.where(frame[dst_col].astype(str).str.strip() == client_ip, 1, -1).astype(np.int8)
    times = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(dtype=np.float64)
    lengths = pd.to_numeric(frame[length_col], errors="coerce").to_numpy(dtype=np.float64)
    keep = np.isfinite(times) & np.isfinite(lengths) & (lengths >= 0)
    times, directions, lengths = times[keep], directions[keep], lengths[keep]
    order = np.argsort(times, kind="stable")
    return PacketTrace(times[order], directions[order], lengths[order].astype(np.float32))


def load_npz_trace(path: str | Path) -> PacketTrace:
    with np.load(path, allow_pickle=False) as data:
        def pick(*names: str) -> np.ndarray:
            for name in names:
                if name in data:
                    return np.asarray(data[name])
            raise ValueError(f"{path} is missing one of {names}")

        times = pick("timestamps", "timestamp", "time").astype(np.float64)
        directions = pick("directions", "direction", "dir").astype(np.int8)
        lengths = pick("lengths", "length", "packet_length").astype(np.float32)
    order = np.argsort(times, kind="stable")
    return PacketTrace(times[order], directions[order], lengths[order])


def load_pcap_trace(path: str | Path, client_ip: str, on_wire: bool = True) -> PacketTrace:
    if not client_ip:
        raise ValueError("PCAP parsing requires client_ip in sessions.csv")
    try:
        import dpkt
    except ImportError as exc:
        raise RuntimeError("Install dpkt to read PCAP traces") from exc
    client = ipaddress.ip_address(client_ip)
    timestamps: list[float] = []
    directions: list[int] = []
    lengths: list[int] = []
    with Path(path).open("rb") as stream:
        try:
            reader = dpkt.pcap.Reader(stream)
        except (ValueError, dpkt.dpkt.NeedData):
            stream.seek(0)
            reader = dpkt.pcapng.Reader(stream)
        for timestamp, raw in reader:
            try:
                ethernet = dpkt.ethernet.Ethernet(raw)
                packet = ethernet.data
                if not isinstance(packet, (dpkt.ip.IP, dpkt.ip6.IP6)):
                    continue
                src = ipaddress.ip_address(packet.src)
                dst = ipaddress.ip_address(packet.dst)
                if src != client and dst != client:
                    continue
                transport = packet.data
                payload_length = len(getattr(transport, "data", b""))
                length = len(raw) if on_wire else payload_length
                timestamps.append(float(timestamp))
                directions.append(1 if dst == client else -1)
                lengths.append(length)
            except (ValueError, socket.error, dpkt.dpkt.UnpackError):
                continue
    return PacketTrace(
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(directions, dtype=np.int8),
        np.asarray(lengths, dtype=np.float32),
    )


def load_trace(record: SessionRecord, on_wire: bool = True) -> PacketTrace:
    path = Path(record.trace_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trace file not found for {record.session_id}: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        trace = load_npz_trace(path)
    elif suffix == ".csv":
        trace = load_csv_trace(path, record.client_ip, prefer_payload_length=not on_wire)
    elif suffix in {".pcap", ".pcapng"}:
        trace = load_pcap_trace(path, record.client_ip, on_wire=on_wire)
    else:
        raise ValueError(f"Unsupported trace format {suffix}: {path}")
    scale = float(record.timestamp_scale or 1.0)
    if scale <= 0:
        raise ValueError(f"timestamp_scale must be positive for {record.session_id}")
    if scale != 1.0:
        trace = PacketTrace(trace.timestamps / scale, trace.directions, trace.lengths)
    return trace


def stable_session_id(path: str | Path, root: str | Path) -> str:
    relative = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    return hashlib.sha1(relative.encode("utf-8")).hexdigest()[:16]


def read_records(path: str | Path) -> list[SessionRecord]:
    table = pd.read_csv(path, keep_default_na=False)
    return [SessionRecord.from_dict(row) for row in table.to_dict(orient="records")]


def write_records(path: str | Path, records: Iterable[SessionRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.to_dict() for record in records]
    if not rows:
        raise ValueError("Refusing to write an empty session index")
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

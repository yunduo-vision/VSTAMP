from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from .abr_modes import fit_abr_modes, quality_trajectory
from .cache import load_feature_cache, save_feature_cache
from .common import SessionRecord, load_trace, write_records
from .longenough import discover_longenough
from .packet_features import NormalizationStats, aggregate_multiscale
from .ydms import discover_ydms


def _deduplicate(records: Iterable[SessionRecord]) -> list[SessionRecord]:
    seen: set[tuple[str, str]] = set()
    output: list[SessionRecord] = []
    for record in records:
        key = (record.video_id, record.duplicate_group or record.session_id)
        if key in seen:
            continue
        seen.add(key)
        output.append(record)
    return output


def _filter_min_sessions(records: list[SessionRecord], minimum: int) -> list[SessionRecord]:
    counts = Counter(record.video_id for record in records)
    return [record for record in records if counts[record.video_id] >= minimum]


def assign_identity_splits(
    records: list[SessionRecord],
    train_count: int,
    validation_count: int,
    test_count: int,
    seed: int,
) -> dict[str, list[str]]:
    identities = sorted({record.video_id for record in records})
    expected = train_count + validation_count + test_count
    if len(identities) != expected:
        raise ValueError(f"Expected exactly {expected} usable identities, found {len(identities)}")
    group_to_identities: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.duplicate_group:
            group_to_identities[record.duplicate_group].add(record.video_id)
    parent = {identity: identity for identity in identities}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for linked in group_to_identities.values():
        linked_list = sorted(linked)
        for other in linked_list[1:]:
            union(linked_list[0], other)
    components: dict[str, list[str]] = defaultdict(list)
    for identity in identities:
        components[find(identity)].append(identity)
    rng = np.random.default_rng(seed)
    blocks = list(components.values())
    rng.shuffle(blocks)
    targets = {"train": train_count, "validation": validation_count, "test": test_count}
    splits: dict[str, list[str]] = {name: [] for name in targets}
    for block in sorted(blocks, key=len, reverse=True):
        feasible = [name for name, target in targets.items() if len(splits[name]) + len(block) <= target]
        if not feasible:
            raise ValueError("Duplicate groups prevent the requested exact identity split")
        destination = max(feasible, key=lambda name: targets[name] - len(splits[name]))
        splits[destination].extend(block)
    if any(len(splits[name]) != target for name, target in targets.items()):
        raise ValueError(f"Could not satisfy exact split sizes: { {k: len(v) for k, v in splits.items()} }")
    for name in splits:
        splits[name].sort()
    return splits


def prepare_dataset(
    dataset: str,
    data_root: str | Path,
    output_root: str | Path,
    seed: int = 0,
    strict_counts: bool = True,
    held_out_bandwidth: str = "",
) -> dict[str, int]:
    """Prepare a real LongEnough or YDMS dataset for VSTAMP training."""
    name = dataset.lower()
    if name == "longenough":
        records = discover_longenough(data_root)
        expected_splits = (50, 20, 30)
        minimum_sessions = 1
        on_wire = True
    elif name == "ydms":
        records = discover_ydms(data_root)
        records = _deduplicate(records)
        expected_splits = (96, 38, 58)
        minimum_sessions = 11
        on_wire = False
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if name == "longenough" and strict_counts:
        missing_qoe = [
            record.session_id
            for record in records
            if not record.qoe_path or not Path(record.qoe_path).is_file()
        ]
        if missing_qoe:
            raise ValueError(
                "Formal LongEnough preprocessing requires qoe_path for every session so ABR modes can be "
                f"fit from training identities; missing for {len(missing_qoe)} sessions."
            )
    if name == "ydms" and strict_counts:
        missing_modes = [record.session_id for record in records if record.abr_mode == "unknown"]
        if missing_modes:
            raise ValueError(
                "Formal YDMS preprocessing requires application-state-derived abr_mode metadata; "
                f"missing for {len(missing_modes)} sessions."
            )
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_cache = output / "raw_cache"
    raw_cache.mkdir(exist_ok=True)
    usable: list[SessionRecord] = []
    for record in tqdm(records, desc=f"Aggregating {dataset}"):
        trace = load_trace(record, on_wire=on_wire)
        if trace.duration < 60.0:
            continue
        features, masks = aggregate_multiscale(trace)
        cache_path = raw_cache / f"{record.session_id}.npz"
        save_feature_cache(cache_path, features, masks)
        record.extra["raw_cache"] = str(cache_path)
        usable.append(record)
    if not usable:
        raise ValueError("No sessions contain at least 60 seconds of packet traffic")
    usable = _filter_min_sessions(usable, minimum_sessions)
    retained_cache = {Path(record.extra["raw_cache"]) for record in usable}
    for cache_path in raw_cache.glob("*.npz"):
        if cache_path not in retained_cache:
            cache_path.unlink()
    identities = sorted({record.video_id for record in usable})
    expected_total = sum(expected_splits)
    if strict_counts and len(identities) != expected_total:
        raise ValueError(
            f"{dataset} requires {expected_total} usable identities after 60-second filtering; "
            f"found {len(identities)}. Check the dataset subset and sessions.csv metadata."
        )
    if name == "longenough" and strict_counts:
        bandwidth_counts = Counter((record.video_id, record.bandwidth) for record in usable)
        expected_bandwidths = {"bw1", "bw2", "bw4", "bw8"}
        observed_bandwidths = {record.bandwidth for record in usable}
        if observed_bandwidths != expected_bandwidths:
            raise ValueError(
                f"Formal LongEnough preprocessing requires {sorted(expected_bandwidths)}; "
                f"found {sorted(observed_bandwidths)}"
            )
        invalid = {
            f"{identity}/{bandwidth}": bandwidth_counts[(identity, bandwidth)]
            for identity in identities
            for bandwidth in expected_bandwidths
            if bandwidth_counts[(identity, bandwidth)] != 10
        }
        if invalid:
            raise ValueError(
                "Formal LongEnough preprocessing requires 10 usable sessions per video-bandwidth; "
                f"mismatches include {dict(list(invalid.items())[:5])}"
            )
    if len(identities) < 3:
        raise ValueError("At least three usable identities are required")
    if strict_counts:
        split_counts = expected_splits
    else:
        train = max(1, int(round(0.5 * len(identities))))
        validation = max(1, int(round(0.2 * len(identities))))
        split_counts = (train, validation, len(identities) - train - validation)
    splits = assign_identity_splits(usable, *split_counts, seed=seed)
    identity_split = {identity: split for split, ids in splits.items() for identity in ids}
    for record in usable:
        record.split = identity_split[record.video_id]
    training_features = [
        load_feature_cache(record.extra["raw_cache"])[0]
        for record in usable
        if record.split == "train"
        and (not held_out_bandwidth or record.bandwidth != held_out_bandwidth)
    ]
    if not training_features:
        raise ValueError("No training features remain after held-out-bandwidth filtering")
    stats = NormalizationStats.fit(training_features, (100, 500, 2000))
    stats.save(output / "normalization_stats.npz")

    feature_root = output / "features"
    for record in tqdm(usable, desc="Applying training-only normalization"):
        cache_path = Path(record.extra.pop("raw_cache"))
        features, masks = load_feature_cache(cache_path)
        normalized = stats.transform(features)
        feature_path = feature_root / f"{record.session_id}.npz"
        save_feature_cache(feature_path, normalized, masks)
        record.feature_path = str(feature_path)
    for cache_path in raw_cache.glob("*.npz"):
        cache_path.unlink()
    raw_cache.rmdir()

    if name == "longenough":
        trajectories: dict[str, np.ndarray] = {}
        for record in usable:
            if record.qoe_path and Path(record.qoe_path).is_file():
                trajectories[record.session_id] = quality_trajectory(record.qoe_path)
        if strict_counts and len(trajectories) != len(usable):
            missing = len(usable) - len(trajectories)
            raise ValueError(
                f"Formal LongEnough preprocessing requires a readable quality trajectory for every session; "
                f"{missing} are missing. Supply qoe_path in sessions.csv."
            )
        train_trajectories = [
            trajectories[record.session_id]
            for record in usable
            if record.split == "train"
            and record.session_id in trajectories
            and (not held_out_bandwidth or record.bandwidth != held_out_bandwidth)
        ]
        if len(train_trajectories) >= 11:
            abr_model, silhouette = fit_abr_modes(train_trajectories, seed=seed)
            abr_model.save(output / "abr_cluster_model.pkl")
            with (output / "abr_silhouette.json").open("w", encoding="utf-8") as stream:
                json.dump({str(k): value for k, value in silhouette.items()}, stream, indent=2)
            with_trajectory = [record for record in usable if record.session_id in trajectories]
            labels = abr_model.predict(np.stack([trajectories[record.session_id] for record in with_trajectory]))
            for record, label in zip(with_trajectory, labels):
                record.abr_mode = label
        elif strict_counts:
            raise ValueError("Not enough training quality trajectories to fit ABR modes")

    if name == "ydms" and strict_counts:
        missing_modes = [record.session_id for record in usable if record.abr_mode == "unknown"]
        if missing_modes:
            raise ValueError(
                "Formal YDMS preprocessing requires application-state-derived ABR modes; "
                f"missing for {len(missing_modes)} sessions. Supply abr_mode or qoe_path in sessions.csv."
            )

    write_records(output / "sessions.csv", usable)
    with (output / "splits.json").open("w", encoding="utf-8") as stream:
        json.dump(splits, stream, indent=2, sort_keys=True)
    summary = {
        "sessions": len(usable),
        "identities": len({record.video_id for record in usable}),
        "train_identities": len(splits["train"]),
        "validation_identities": len(splits["validation"]),
        "test_identities": len(splits["test"]),
    }
    with (output / "preprocessing_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "dataset": dataset,
                "seed": seed,
                "held_out_bandwidth": held_out_bandwidth,
                "summary": summary,
            },
            stream,
            indent=2,
        )
    return summary

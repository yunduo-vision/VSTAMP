from pathlib import Path

import pandas as pd

from vstamp.data.cache import load_feature_cache
from vstamp.data.common import read_records
from vstamp.data.preprocess import prepare_dataset


def test_csv_to_processed_feature_path(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for video in range(3):
        video_dir = raw / f"video_{video}"
        video_dir.mkdir(parents=True)
        for session in range(2):
            pd.DataFrame(
                {
                    "timestamp": [1000.0, 1000.1, 1030.0, 1060.1],
                    "direction": ["downstream", "upstream", "downstream", "upstream"],
                    "length": [100 + video, 50 + session, 200 + video, 60],
                }
            ).to_csv(video_dir / f"bw1_session_{session}.csv", index=False)
    processed = tmp_path / "processed"
    summary = prepare_dataset(
        "longenough", raw, processed, seed=4, strict_counts=False
    )
    assert summary["identities"] == 3
    assert summary["sessions"] == 6
    records = read_records(processed / "sessions.csv")
    features, masks = load_feature_cache(records[0].feature_path)
    assert features[100].shape == (600, 5)
    assert masks[2000].shape == (30,)
    assert (processed / "normalization_stats.npz").is_file()
    assert (processed / "splits.json").is_file()


# VSTAMP Dataset Download and Preprocessing Guide

This standalone document explains how to obtain, organize, validate, preprocess, and load the LongEnough and YDMS datasets used by VSTAMP. It contains all data-related instructions needed to prepare the experiments without relying on the repository's main `README.md`.

> The datasets are not distributed with this repository. Download and use them only for purposes permitted by their publishers, and follow all citation, licensing, and privacy requirements included with each release.

## 1. Processing objective

The preprocessing pipeline converts packet-level video streaming sessions into fixed 60-second, multiscale traffic features. Video identities are split into disjoint training, validation, and test sets.

| Dataset | Identities in the formal task | Identity split (train/validation/test) | Packet-length definition | Main use |
|---|---:|---:|---|---|
| LongEnough | 100 | 50 / 20 / 30 | Full on-wire packet length | Cross-ABR-mode, cross-bandwidth, and unseen-bandwidth evaluation |
| YDMS | 192 | 96 / 38 / 58 | Transport-layer payload length | Random 5-shot and cross-state evaluation |

An identity is a `video_id`, not a viewer, device, or individual playback session. A video identity normally has multiple playback sessions collected under different network and adaptive streaming conditions.

The pipeline enforces the following leakage controls:

- Identities are split before training, validation, and test sessions are consumed. The same video cannot occur in multiple splits.
- Feature means and standard deviations are fitted on training identities only.
- The LongEnough ABR clustering model is fitted on quality trajectories from training identities only.
- An unseen-bandwidth run also excludes the held-out bandwidth from training feature statistics and ABR clustering.
- Test identities are never used to select preprocessing parameters, model settings, or checkpoints.

## 2. Official datasets and download links

### 2.1 LongEnough

LongEnough accompanies *Raising the Bar: Improved Fingerprinting Attacks and Defenses for Video Streaming Traffic*. The release contains undefended, defended, and variable-bandwidth traffic traces together with QoE information.

Official resources:

- Project and dataset description: [trafnex/raising-the-bar](https://github.com/trafnex/raising-the-bar)
- Publisher-provided data folder: [LongEnough official shared folder](https://liuonline-my.sharepoint.com/:f:/g/personal/davha914_student_liu_se/ErK6esYd5IdOiuvfLnXK6NoBEdlj579MlXBvG2wkfQEozg?e=sCHtWp)

The official project page lists these archives:

| Archive | Compressed size | Expanded size | Use in this repository |
|---|---:|---:|---|
| `LongEnough.zip` | 18 GB | 81 GB | Default undefended data; not the preferred input for the main VSTAMP task |
| `LongEnough-defended.zip` | 16 GB | 71 GB | Defended traces; not used by the default VSTAMP configuration |
| `LongEnough-variable.zip` | 21 GB | 83 GB | **Required for the formal VSTAMP LongEnough task** |

Download and extract `LongEnough-variable.zip`. The formal reproduction uses the undefended, offset-0, variable-bandwidth subset with `bw1`, `bw2`, `bw4`, and `bw8` conditions.

The preprocessing code does not infer whether an additional directory is defended or belongs to a nonzero offset. Do not place multiple experimental subsets below the same `--data-root`. If the release layout is ambiguous, use the explicit `sessions.csv` interface described in Section 4.

After duration filtering, formal preprocessing requires:

- exactly 100 video identities;
- exactly 10 usable sessions per identity and bandwidth condition;
- all four bandwidth labels: `bw1`, `bw2`, `bw4`, and `bw8`;
- 4,000 usable sessions in total; and
- a readable quality/QoE trajectory for every session so that ABR modes can be constructed without test leakage.

### 2.2 YDMS

YDMS, the YouTube Dataset on Mobile Streaming, contains synchronized network-, transport-, and application-layer measurements from the native YouTube mobile client. The complete release contains 11,142 measurements, 246 videos, and 1,081.18 hours of playback. These values describe the full public dataset and should not be confused with the 192 identities retained by the VSTAMP filtering protocol.

Official resources:

- Dataset download: [Figshare dataset, version 2](https://doi.org/10.6084/m9.figshare.19096823.v2). The page provides a **Download all** option and reports a compressed download size of approximately 3.21 GB.
- Dataset paper: [Scientific Data 9, 293 (2022)](https://www.nature.com/articles/s41597-022-01418-y)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated on the Figshare page

An official measurement directory normally contains:

```text
<measurement>/
  all_network_traffic.csv   # All observed network traffic; not read by default
  video_traffic.csv         # Extracted video traffic; used as trace_path
  stats_for_nerds.csv       # Raw YouTube Stats for Nerds information
  application_data.csv      # Processed application and QoE information
  bw_setting.csv            # Bandwidth settings over time
```

The adapter recursively discovers `video_traffic.csv`, infers the video identity from `application_data.csv` or `stats_for_nerds.csv`, and derives a session mode from `application_data.csv`. It then removes sessions shorter than 60 seconds and identities with fewer than 11 usable sessions. Formal preprocessing requires exactly 192 identities after these filters.

## 3. Download verification and storage layout

Keep raw datasets outside the Git repository to avoid accidentally committing large files:

```text
D:/datasets/
  LongEnough-variable/
  YDMS/

vstamp/
  data/
    processed/
      longenough/
      ydms/
```

LongEnough expands to approximately 83 GB. Additional space is required for processed features and temporary caches. At least 120 GB of free space is a practical starting point, but this is an operational recommendation rather than an official minimum.

Record the archive size and a local SHA-256 digest after downloading. If the publisher does not provide a checksum, the local digest is still useful for verifying copies shared within a team.

Windows PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 D:\Downloads\LongEnough-variable.zip
Get-FileHash -Algorithm SHA256 D:\Downloads\YDMS.zip
```

Linux or macOS:

```bash
sha256sum ~/Downloads/LongEnough-variable.zip
sha256sum ~/Downloads/YDMS.zip
```

Preserve the dataset README and license files included in each archive. The BSD-3-Clause license in the LongEnough code repository applies to that repository's code and must not automatically be assumed to cover every data file. Use the terms distributed with the dataset itself.

## 4. Input discovery and `sessions.csv`

Both adapters support two input methods:

1. **Automatic discovery:** recursively scan a compatible official directory layout.
2. **Explicit manifest, recommended:** place `sessions.csv` at the top level of `--data-root`.

Automatic discovery is useful for an initial inspection. A manifest is preferred for formal experiments because it fixes the interpretation of identity, bandwidth, direction, duplicate relationships, and QoE files even if a release layout changes.

### 4.1 LongEnough automatic discovery

The adapter recursively accepts `.pcap`, `.pcapng`, `.csv`, and `.npz` trace files. It infers a bandwidth from `bw1`, `bw2`, `bw4`, or `bw8` in the path and uses the first non-condition directory as the `video_id`.

A CSV in the same directory is considered a candidate QoE companion when its filename contains one of the following strings:

```text
qoe, quality, playback, representation, bitrate
```

The following is only an illustrative compatible layout. Consult the README included with the downloaded dataset for the actual release structure.

```text
LongEnough-variable/
  video_000/
    bw1/
      session_00.pcap
      session_00_quality.csv
    bw2/
    bw4/
    bw8/
  video_001/
  ...
```

Use a manifest when one QoE file applies to multiple traces, a directory name does not uniquely identify a video, or a PCAP requires a client IP address.

### 4.2 YDMS automatic discovery

The adapter recursively searches for `video_traffic.csv`. It reads up to the first 100 rows of `application_data.csv` and `stats_for_nerds.csv`, then uses the modal value from the first recognized video identity column.

Supported video identity columns, matched case-insensitively:

```text
videoid, video_id, video id, content_id
```

Supported application-state columns:

```text
phase, player_state, buffer_state, state, playback_state
```

The current YDMS state-to-mode rules are:

- `starved` if at least 20% of states contain `starv`, `stall`, `rebuffer`, or `deplet`;
- otherwise `high_stable` if at least 80% contain `steady`, `stable`, or `normal`;
- otherwise `transitional`.

If the official files use different column names, set `abr_mode` directly in a manifest or provide a compatible application-state CSV through `qoe_path`.

### 4.3 Manifest fields

Required `sessions.csv` columns:

| Column | Type | Description |
|---|---|---|
| `session_id` | String | Globally unique, stable session identifier; do not use an unstable row number |
| `video_id` | String | Video identity; all sessions of the same content must use the same value |
| `trace_path` | Path | PCAP/PCAPNG, packet CSV, or NPZ; relative paths are resolved from `--data-root` |

Recommended columns:

| Column | Default | Description |
|---|---|---|
| `bandwidth` | `unknown` | Use `bw1`, `bw2`, `bw4`, or `bw8` for LongEnough |
| `abr_mode` | `unknown` | Existing mode label; formal LongEnough preprocessing overwrites it using training-only QoE clustering |
| `duplicate_group` | Empty | Shared identifier for duplicate or linked sessions |
| `client_ip` | Empty | Required for PCAP and CSV traces without an explicit direction column |
| `qoe_path` | Empty | LongEnough quality trajectory or YDMS application-state file |
| `timestamp_scale` | `1.0` | Timestamp units per second: `1` for seconds, `1000` for milliseconds, or `1000000` for microseconds |
| `extra` | `{}` | JSON text for evaluation-only metadata |

The `split` and `feature_path` columns are produced by preprocessing. They are not needed in a raw manifest. If present, the identity split and feature path are regenerated.

Example LongEnough manifest:

```csv
session_id,video_id,trace_path,bandwidth,client_ip,qoe_path,duplicate_group,timestamp_scale,extra
le_v000_bw1_00,video_000,video_000/bw1/session_00.pcap,bw1,192.0.2.10,video_000/bw1/session_00_quality.csv,le_v000_bw1_00,1,"{}"
le_v000_bw2_00,video_000,video_000/bw2/session_00.pcap,bw2,192.0.2.10,video_000/bw2/session_00_quality.csv,le_v000_bw2_00,1,"{}"
```

Example YDMS manifest:

```csv
session_id,video_id,trace_path,abr_mode,qoe_path,duplicate_group,timestamp_scale,extra
ydms_000001,video_A,measurements/000001/video_traffic.csv,starved,measurements/000001/application_data.csv,run_000001,1,"{}"
ydms_000002,video_A,measurements/000002/video_traffic.csv,high_stable,measurements/000002/application_data.csv,run_000002,1,"{}"
```

The IP address and IDs above are placeholders. Replace them with metadata from the actual dataset.

### 4.4 Duplicate sessions

YDMS deduplication uses `(video_id, duplicate_group)` as its key. If `duplicate_group` is empty, the adapter falls back to `session_id`.

Automatic discovery generates a distinct session ID for every file, so it cannot identify near-duplicates from content alone. Known duplicate relationships must be encoded explicitly by assigning the same `duplicate_group` in the manifest.

If the same duplicate group links multiple video identities, the identity splitter treats those identities as an inseparable component to prevent leakage. Preprocessing fails explicitly if such components make the exact 96/38/58 or 50/20/30 split impossible.

## 5. Packet-trace formats

### 5.1 CSV

A packet CSV must contain a timestamp, a packet length, and one of these direction representations:

- an explicit direction column; or
- source and destination IP columns together with `client_ip` in the manifest.

Supported column aliases, matched case-insensitively:

| Purpose | Accepted columns |
|---|---|
| Timestamp | `timestamp`, `time`, `ts`, `frame.time_epoch`, `relative_timestamp` |
| Explicit direction | `direction`, `dir`, `packet_direction` |
| LongEnough length priority | `length`, `packet_length`, `frame.len`, `payload_length`, `payloadLength`, `udp.length`, `tcp.len` |
| YDMS payload-length priority | `payload_length`, `payloadLength`, `tcp.len`, `udp.length`, `length`, `packet_length`, `frame.len` |
| Source IP | `src_ip`, `source`, `ip.src`, `src` |
| Destination IP | `dst_ip`, `destination`, `ip.dst`, `dst` |

Direction conventions:

- `+1`, `down`, `downstream`, `in`, or `incoming`: downstream, server to client;
- `-1`, `up`, `upstream`, `out`, or `outgoing`: upstream, client to server.

Minimal example:

```csv
timestamp,direction,length
0.000,+1,1514
0.012,-1,74
0.025,+1,1514
```

Rows are sorted by timestamp. Non-numeric timestamps, non-numeric lengths, and negative lengths are discarded. For millisecond timestamps, set `timestamp_scale=1000`. An incorrect scale usually causes the session to appear shorter than 60 seconds or makes nearly all packets fall into one observation bin.

For a faithful LongEnough reproduction, provide a true on-wire length such as `frame.len`. A payload-only column may be accepted as a fallback but does not reproduce the intended length definition.

### 5.2 NPZ

An NPZ trace must contain three equally sized one-dimensional arrays:

```python
import numpy as np

np.savez_compressed(
    "trace.npz",
    timestamps=np.asarray([...], dtype=np.float64),
    directions=np.asarray([...], dtype=np.int8),  # -1 or +1 only
    lengths=np.asarray([...], dtype=np.float32),
)
```

Accepted key aliases:

- timestamps: `timestamps`, `timestamp`, or `time`;
- directions: `directions`, `direction`, or `dir`;
- lengths: `lengths`, `length`, or `packet_length`.

### 5.3 PCAP and PCAPNG

PCAP input requires the `dpkt` dependency and a `client_ip` value in the manifest. The parser retains IPv4 and IPv6 packets whose source or destination matches the client, then assigns direction from the client's perspective.

- LongEnough uses captured frame length, `len(raw)`, as on-wire length.
- YDMS uses TCP or UDP transport payload length when supplied as PCAP.

Malformed or unsupported individual packets are skipped. A session is filtered if the remaining valid packet span is shorter than 60 seconds.

### 5.4 QoE and quality trajectories

For LongEnough, `qoe_path` must point to a CSV containing both a time column and a numeric quality column.

Accepted time columns:

```text
timestamp, time, relative_timestamp, playback_time, t
```

Accepted quality columns:

```text
bitrate, selected_bitrate, requested_bitrate, quality,
quality_index, resolution, video_quality
```

Each session is resampled into a 30-point trajectory spanning 60 seconds. The trajectories from training identities are standardized and clustered with K-Means. Candidate values `K=2..10` are compared using silhouette score.

When the selected value is `K=3`, clusters are ordered by mean quality and named:

```text
low_or_late, transitional, high_stable
```

If the selected K is not 3, labels are written as `mode_0`, `mode_1`, and so on. These labels are incompatible with the three canonical labels required by the default LongEnough `cross_mode` protocol. After formal preprocessing, inspect `abr_silhouette.json` and the `abr_mode` distribution in the processed `sessions.csv`.

## 6. Environment setup

Run all commands from the repository root.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Confirm that both preparation entry points are available:

```bash
python scripts/prepare_longenough.py --help
python scripts/prepare_ydms.py --help
```

## 7. Run preprocessing

### 7.1 Formal LongEnough data

Windows PowerShell:

```powershell
python scripts/prepare_longenough.py `
  --data-root "D:\datasets\LongEnough-variable" `
  --output-root "data\processed\longenough" `
  --seed 0
```

Linux or macOS:

```bash
python scripts/prepare_longenough.py \
  --data-root /data/LongEnough-variable \
  --output-root data/processed/longenough \
  --seed 0
```

### 7.2 Formal YDMS data

Windows PowerShell:

```powershell
python scripts/prepare_ydms.py `
  --data-root "D:\datasets\YDMS" `
  --output-root "data\processed\ydms" `
  --seed 0
```

Linux or macOS:

```bash
python scripts/prepare_ydms.py \
  --data-root /data/YDMS \
  --output-root data/processed/ydms \
  --seed 0
```

A successful formal LongEnough run prints a summary similar to:

```json
{
  "sessions": 4000,
  "identities": 100,
  "train_identities": 50,
  "validation_identities": 20,
  "test_identities": 30
}
```

The YDMS session count depends on how many sessions remain for each identity after filtering. It is not fixed by the protocol. Formal preprocessing fixes only the identity count and the 96/38/58 identity split.

### 7.3 Small development subsets

The formal identity-count check can be relaxed for code debugging:

```bash
python scripts/prepare_longenough.py \
  --data-root /path/to/small_subset \
  --output-root data/processed/longenough_dev \
  --allow-nonstandard-identity-count
```

This mode uses approximately 50% of identities for training, 20% for validation, and the remainder for testing. It does not synthesize missing data and does not reproduce the paper protocol. At least three usable identities are required. LongEnough still needs enough training quality trajectories to construct meaningful ABR labels.

### 7.4 Unseen-bandwidth preprocessing

Each held-out bandwidth must be processed into a separate output directory so that normalization and ABR fitting remain leakage-free:

```bash
python scripts/prepare_longenough.py \
  --data-root /path/to/LongEnough-variable \
  --output-root data/processed/longenough_unseen_bw1 \
  --seed 0 \
  --held-out-bandwidth bw1

python scripts/prepare_longenough.py \
  --data-root /path/to/LongEnough-variable \
  --output-root data/processed/longenough_unseen_bw8 \
  --seed 0 \
  --held-out-bandwidth bw8
```

The held-out bandwidth is excluded from training feature statistics and training-only ABR clustering. Its features are still generated for evaluation. Train with `configs/longenough_unseen_bw1.yaml` or `configs/longenough_unseen_bw8.yaml`; these configurations also exclude the corresponding bandwidth from training episodes.

## 8. Exact preprocessing sequence

A complete run performs these operations:

1. Discover sessions from the official layout or `sessions.csv`.
2. Deduplicate YDMS sessions using `(video_id, duplicate_group)`.
3. Read, sort, and rescale packet timestamps.
4. Remove sessions with less than 60 seconds of valid packet traffic.
5. For YDMS, remove video identities with fewer than 11 usable sessions.
6. Validate formal identity counts. LongEnough additionally validates all four bandwidths and 10 sessions per identity and bandwidth.
7. Split video identities deterministically using the requested seed.
8. Aggregate the first 60 seconds after the first valid packet at three temporal resolutions.
9. Fit per-resolution, per-feature means and standard deviations using training identities only.
10. Normalize every split and save compressed feature caches.
11. For LongEnough, fit ABR clusters on training-identity QoE trajectories and label all sessions with readable trajectories.
12. Write the session index, identity splits, statistics, and preprocessing manifest.

Each time bin contains five raw features in this order:

1. downstream bytes;
2. upstream bytes;
3. downstream packet count;
4. upstream packet count;
5. mean downstream packet length.

The pipeline applies `log(1+x)` before training-only z-score normalization.

| Resolution | Bins in 60 seconds | Per-session array shape |
|---:|---:|---|
| 100 ms | 600 | `[600, 5]` |
| 500 ms | 120 | `[120, 5]` |
| 2000 ms | 30 | `[30, 5]` |

## 9. Output layout and formats

Typical output:

```text
data/processed/longenough/
  sessions.csv
  splits.json
  normalization_stats.npz
  preprocessing_manifest.json
  abr_cluster_model.pkl       # LongEnough when enough quality trajectories exist
  abr_silhouette.json         # LongEnough when enough quality trajectories exist
  features/
    <session_id>.npz
    ...
```

| File | Contents |
|---|---|
| `sessions.csv` | Filtered sessions, final split, ABR labels, raw paths, and absolute feature paths |
| `splits.json` | Mapping from `train`, `validation`, and `test` to video identity lists |
| `normalization_stats.npz` | `mean_100/std_100`, `mean_500/std_500`, and `mean_2000/std_2000` |
| `features/*.npz` | `x_100/m_100`, `x_500/m_500`, and `x_2000/m_2000` |
| `preprocessing_manifest.json` | Dataset name, seed, held-out bandwidth, and summary counts |
| `abr_cluster_model.pkl` | Training-only scaler, K-Means model, and label mapping |
| `abr_silhouette.json` | Silhouette score for each tested K |

The `x_*` arrays are `float32` features. The `m_*` arrays are `float32` validity masks. A complete 60-second cache has an all-one mask. Query-truncation evaluation reconstructs a partially visible mask from the raw trace.

The processed `sessions.csv` stores absolute `trace_path` and `feature_path` values. Do not move the raw or processed directories after preparation. If migration is necessary, update every stored path consistently or rerun preprocessing.

## 10. Load processed data in Python

```python
from vstamp.data.dataset import ProcessedDataset

dataset = ProcessedDataset(
    root="data/processed/longenough",
    split="train",            # train, validation, test, or None
    cache_size=256,
)

records = dataset.records[:4]
session_ids = [record.session_id for record in records]
features, masks = dataset.tensors(session_ids, device="cpu")

for resolution in sorted(features):
    print(
        resolution,
        features[resolution].shape,  # [batch, bins, 5]
        masks[resolution].shape,     # [batch, bins]
    )
```

Load a partial observation:

```python
features, masks = dataset.tensors(
    session_ids,
    device="cpu",
    retained_fraction=0.30,   # Keep the first 18 seconds
)
```

Partial observations are rebuilt from the raw trace, normalized with the saved training statistics, and zeroed outside the visible interval. The original data must remain accessible because this operation uses `trace_path` from the processed manifest.

Training configurations select the processed directory through `data.processed_root`:

```yaml
dataset:
  name: longenough
data:
  processed_root: data/processed/longenough
  observation_seconds: 60
  resolutions_ms: [100, 500, 2000]
```

Start training with:

```bash
python scripts/train.py --config configs/longenough.yaml --seed 0
python scripts/train.py --config configs/ydms.yaml --seed 0
```

## 11. Additional metadata for real-overlap evaluation

The `real_overlap` protocol must not infer ground-truth overlap from network traffic. Every evaluated session must contain media interval metadata derived from player progress and buffer logs in its `extra` JSON object:

```json
{"media_start": 12.5, "media_end": 70.2}
```

These values are evaluation-only and are not passed to the feature encoder or training pipeline. In a CSV cell, escape the JSON according to CSV quoting rules:

```csv
extra
"{""media_start"": 12.5, ""media_end"": 70.2}"
```

Do not run or report `real_overlap` results without reliable player-side timing information.

## 12. Post-processing integrity checks

### 12.1 Inspect the summary

```powershell
Get-Content data\processed\longenough\preprocessing_manifest.json
Get-Content data\processed\longenough\splits.json
```

Formal LongEnough data must report 100 identities and a 50/20/30 split. Formal YDMS data must report 192 identities and a 96/38/58 split.

### 12.2 Inspect manifest distributions

```python
from pathlib import Path

import pandas as pd

df = pd.read_csv("data/processed/longenough/sessions.csv", keep_default_na=False)
print(df.groupby("split")["video_id"].nunique())
print(df.groupby(["bandwidth", "abr_mode"]).size())
print("duplicate session_id:", df["session_id"].duplicated().sum())
print("missing feature files:", (~df["feature_path"].map(lambda p: Path(p).is_file())).sum())
```

### 12.3 Inspect one feature cache

```python
import numpy as np
import pandas as pd

row = pd.read_csv("data/processed/longenough/sessions.csv").iloc[0]
with np.load(row.feature_path, allow_pickle=False) as item:
    print(item.files)
    print(item["x_100"].shape, item["x_500"].shape, item["x_2000"].shape)
    assert np.isfinite(item["x_100"]).all()
```

## 13. Troubleshooting

### `LongEnough root not found` or `YDMS root not found`

Make sure `--data-root` points to the extracted directory rather than the ZIP archive. Quote Windows paths that contain spaces.

### `No ... traces found`

LongEnough requires a supported PCAP, PCAPNG, CSV, or NPZ below the root. YDMS requires a recursively discoverable `video_traffic.csv`. Create `sessions.csv` if filenames or directories differ from the supported layout.

### `PCAP parsing requires client_ip`

Automatic discovery cannot reliably determine which address belongs to the playback client. Create a manifest and assign the correct `client_ip` to each PCAP.

### `has no direction column`

Add a supported direction column to the CSV, or provide `client_ip` and supported source and destination IP columns.

### `No sessions contain at least 60 seconds`

Check the following before changing the protocol:

- timestamp units and `timestamp_scale`;
- whether the exported CSV contains only a short interval;
- whether `client_ip` is correct for PCAP input; and
- whether the archive or extracted files are incomplete.

### `Expected exactly ... usable identities`

Formal counts are checked after duration and minimum-session filtering. Before using the development override, verify that the correct subset was downloaded, `video_id` values were not accidentally split, YDMS identities have at least 11 sessions, and LongEnough does not include unrelated defense or offset conditions.

### `requires 10 usable sessions per video-bandwidth`

At least one LongEnough session is shorter than 60 seconds, missing, assigned the wrong bandwidth, or assigned the wrong identity. The exception reports example mismatches that can be traced back to the manifest.

### `requires qoe_path for every session`

Formal LongEnough processing does not accept sessions without QoE information. Confirm that every file exists and contains a supported time column and numeric quality column.

### `requires application-state-derived abr_mode metadata`

The YDMS application-state column was not recognized. Inspect `application_data.csv` or assign `abr_mode` explicitly in the manifest.

### `Duplicate groups prevent the requested exact identity split`

A duplicate group links too many identities to satisfy the exact split. Review `duplicate_group`; it should encode a genuine duplicate or dependency relationship and must not be one constant shared by the full dataset.

### Loading fails after moving a processed directory

The output manifest contains absolute paths. Restore the original locations, update all paths consistently, or rerun preprocessing. Rerunning is the safest option for a formal reproduction.
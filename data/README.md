# Datasets

The MovieLens datasets are not included. GroupLens does not allow them to be redistributed without permission, so download them from the official site. Their usage terms apply.

## MovieLens-100K (all stages)

This command downloads the dataset if needed, checks its hash, and creates the split:

```bash
python -m src.data
```

It writes `data/raw/ml-100k/`, `data/processed/ml-100k/`, `results/data_summary.json`, and `results/DATASET_REPORT.md`.

| Check | Expected value |
|---|---|
| `u.data` SHA-256 | `06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490` |
| Split fingerprint | `6faed6fc3d47b5b6fa638adfeea83cd7409d50c39fa01f85c10379d0daef6989` |
| Eligible users / items | 942 / 1,682 |
| Train / validation / test | 53,491 / 942 / 942 |

## MovieLens-1M (E3 and E4 only)

`experiments/run_noise_replication.py prepare` reads the archive and its checksum from fixed paths in `/tmp`, so download them there first:

```bash
curl -L -o /tmp/cs670_ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
curl -L -o /tmp/cs670_ml-1m.zip.md5 https://files.grouplens.org/datasets/movielens/ml-1m.zip.md5
python experiments/run_noise_replication.py prepare
```

This step stops if the archive does not match the published MD5, and it writes to `data/extensions/ml-1m_noise_v1/`.

| Check | Expected value |
|---|---|
| Archive MD5 | `c4d9eecfca2ab87c1945afe126590906` |
| Split fingerprint | `e68ef765cc661f5db1864f9f8b2843bee70c7ad261c116421d2c7102ec71bd54` |
| Eligible users / items | 6,035 / 3,706 |
| Training positives | 563,206 |

The MovieLens-1M test targets are never scored in this project.

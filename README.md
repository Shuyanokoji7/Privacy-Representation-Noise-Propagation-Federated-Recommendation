# Privacy, Representation, and Noise Propagation in Federated Recommendation

CS-670 course project by Bhavya Chauhan (230294), Chaudhari Divyesh (230325), Avantika Rohite (230245), and Aaradhya Rohi (230011).

## What the project did

Federated recommendation keeps each user's ratings on their own device, but protecting a whole user with differential privacy (DP) requires clipping updates and adding noise, which can hurt recommendations. This project asks whether a smaller shared item representation improves that trade-off, and how injected noise turns into disturbance in item matrices, scores, and rankings.

The study uses BPR matrix factorization on MovieLens. Stages B0 to B4 build up from centralized training to private low-rank models, and E1 to E4 then investigate how noise behaves:

| Stage | What it tests | Main script | Report section |
|---|---|---|---|
| Phase 0 | Split, evaluator, random and popularity references | `src/data.py`, `src/baselines.py` | 5, 6 |
| B0, B0-UW | Centralized BPR, with a user-weighted control | `experiments/run_b0.py` | 6 |
| B1, B2 | Federated full matrix, without and with DP | `experiments/run_b1.py`, `run_b2.py` | 6 |
| B3, B4 | Federated low-rank factors, without and with DP | `experiments/run_b3.py`, `run_b4.py` | 6 |
| E1 | Full, Two, and FixedB models under equal tuning, plus private popularity | `experiments/run_effective_noise.py` | 7 |
| E2, E2b | Pulse and reset probes of local-state memory, with an aligned noise control | `experiments/probe_noise_memory.py`, `probe_noise_memory_aligned.py` | 8 |
| E3 | MovieLens-1M replication and balancing controls | `experiments/run_noise_replication.py` | 9 |
| E4 | Screen of ranking-tail behavior in item-pair margins | `experiments/probe_ranking_tails.py` | 10 |

Main findings:

- **Low rank is not a clear win.** Private low-rank models beat the private full matrix in four of sixteen adjusted comparisons, and all four nonprivate low-rank variants fall below the converged full model.
- **Fixing a public factor does not help.** It wins no adjusted comparison and loses eight of sixteen to training both factors.
- **Popularity is hard to beat under strong privacy.** A private popularity baseline exceeds every collaborative point estimate at target epsilon of 4 or below.
- **Noise norm alone does not predict harm.** Effective noise depends on factor geometry, local user scale, clipping, and pairwise margins.
- **Two-factor comparisons depend on noise coupling.** Large trajectory differences shrink substantially under an orthogonal alignment control, and the MovieLens-1M runs reproduce this dependence.
- **Non-Gaussian tails do not explain the results.** Across 3,200 item-pair records, a Gaussian approximation's sign-flip probability error is at most 7.67×10⁻⁶.

The [report](docs/CS670_Report.pdf) has the full method and results, the [supplement](docs/CS670_Supplement.pdf) has the full tables, and the [slides](docs/CS670_Presentation.pptx) are the talk.

## Layout

| Path | What it holds |
|---|---|
| `src/` | Data pipeline, models, federated and private training, privacy accounting, evaluation |
| `experiments/` | Runners and analysis scripts for each stage |
| `configs/` | Final settings for B0 to B4 and E1 |
| `docs/extensions/` | Protocols for E1 to E4, which the runners read and hash |
| `scripts/` | Figure generation and the synthetic coupling example |
| `results/`, `figures/` | Final result tables and report figures |
| `data/` | Dataset download steps and expected hashes; see [data/README.md](data/README.md) |
| `tests/` | Unit and protocol tests |

## Setup

Tested with Python 3.10 on CPU. Opacus provides the privacy accountants only and is installed without dependencies so that it cannot replace the pinned PyTorch build.

```bash
python -m venv .venv
source .venv/bin/activate
grep -v '^opacus' requirements.txt > /tmp/requirements-no-opacus.txt
pip install -r /tmp/requirements-no-opacus.txt
pip install --no-deps opacus==1.6.0
```

## Quick check

Run these from this folder. They take a few minutes in total and do not score any test targets.

```bash
python -m src.data                               # download ML-100K, check its hash, create the split
python -m pytest                                 # unit and protocol tests
python experiments/run_b2.py --accounting        # solve the noise multipliers at T = 50
python experiments/run_b2.py --sanity --clip 1   # one private run, validation only, with mechanism checks
python scripts/make_figures.py                   # regenerate Figures 1 to 6 from results/
python scripts/make_followup_figures.py          # regenerate Figures 7 to 9 from results/
```

## Running the full study

Run the stages in this order, because later stages read earlier outputs. Stages marked as scoring test targets evaluate the MovieLens-100K test set.

```bash
# B0 and B0-UW (scores test)
python experiments/run_b0.py --config configs/b0.yaml --seeds 42 123 2026
python experiments/run_b0.py --config configs/b0_uw.yaml --seeds 42 123 2026

# B1 (scores test), then the weighting comparison
python experiments/run_b1.py --config configs/b1.yaml --seeds 42 123 2026
python experiments/compare_weighting.py

# B2 (scores test)
python experiments/run_b2.py --accounting
python experiments/run_b2.py --sweep --seeds 42 123 2026

# B3 (--final-score scores test), including B1 seeds 7 and 99
python experiments/run_b3.py --final-train
python experiments/run_b3.py --final-score
python experiments/run_b3.py --b1-supplemental

# B4 (--score-test scores test), then the B3 and B4 analysis
python experiments/run_b4.py --final-train
python experiments/run_b4.py --select-rank
python experiments/run_b4.py --score-test
python experiments/analyse_b3b4.py

# E1 (score-test scores test)
python experiments/run_effective_noise.py accounting
python experiments/run_effective_noise.py search
python experiments/run_effective_noise.py final-train
python experiments/run_effective_noise.py freeze
python experiments/run_effective_noise.py score-test
python experiments/analyse_effective_noise.py validation
python experiments/analyse_effective_noise.py test

# E2 and E2b (read E1 checkpoints; no held-out labels)
python experiments/probe_noise_memory.py check
python experiments/probe_noise_memory.py run
python experiments/probe_noise_memory.py summary
python experiments/probe_noise_memory_aligned.py check
python experiments/probe_noise_memory_aligned.py

# E3 (needs the MovieLens-1M download in data/README.md; never scores test)
python experiments/run_noise_replication.py prepare
python experiments/run_noise_replication.py check
python experiments/run_noise_replication.py train
python experiments/run_noise_replication.py probe
python experiments/run_noise_replication.py summary

# E4 (reads E1 and E3 checkpoints; no training and no labels)
python experiments/probe_ranking_tails.py check
python experiments/probe_ranking_tails.py run
python experiments/probe_ranking_tails.py summary
python experiments/summarize_noise_followup.py
```

Useful options: most long stages accept `--workers N`. The configs already hold the final settings, so the validation searches that chose them are optional:

- `run_b0.py --search`
- `run_b1.py --search`
- `run_b2.py --t-sweep`, `--clip-search`, and `--server-lr-check`
- `run_b3.py --search`
- `run_b4.py --lr-c-search` and `--slr-search`

## Before a full rerun

- **Outputs:** `results/` holds the final tables from the original runs. The full study was not rerun for this package, so regenerating the figures does not need any training.
- **Missing inputs:** model checkpoints, raw per-run logs, and the datasets are not included. E2, E3, E4, and the analysis scripts need the outputs of the stages they depend on.
- **Extra seeds:** `analyse_b3b4.py` also expects B2 runs for seeds 7 and 99 under `results/raw/supplemental_b2/`, and `analyse_effective_noise.py` reads B4 per-user outputs from `results/raw/b4_runs/`.

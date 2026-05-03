# Dianchi Water: Dataset & MAS Benchmark Code

Anonymous code release for the paper *"A High-Frequency Multi-Station Surface
Water Quality Dataset and Mask-View Augmentation Benchmark for Time-Series
Imputation"*.

This repository provides the **Dianchi Water** dataset, a fully reproducible
**MAS (Mask-view Augmentation Strategy)** training pipeline, and benchmark
scripts for both **imputation** (the paper's main task, 11 architectures) and
**forecasting from incomplete inputs** (Appendix, 7 architectures), with
cross-dataset replication on Beijing Air Quality and ETTh1.

## Repository structure

```
├── cleaning/
│   ├── pipeline.py              # Eight-stage cleaning pipeline (public Dianchi subset)
│   ├── merge_decisions.json     # Station merge/separation decision list
│   ├── manual_corrections.json  # Station name canonicalization rules
│   ├── README.md
│   └── public_raw_dianchi/      # Raw yearly CSVs (input to pipeline)
│       ├── dianchi_raw_2022.csv
│       ├── dianchi_raw_2023.csv
│       └── dianchi_raw_2024.csv
├── mas/
│   ├── data_loader.py           # Auto-download from HuggingFace + reindex to 4h grid
│   ├── mask_generator.py        # point / subseq / block mask generation
│   └── mas_preprocessor.py      # MAS augmented data builder
│                                #   (rolling stride, pattern/rate/seed mixing)
├── benchmark/
│   ├── generate_baseline.py     # Pre-generate the 5 fixed test masks → H5 files
│   ├── generate_aug.py          # Pre-generate MAS-augmented train.h5
│   ├── train_baseline.py        # Standard recipe (single pattern, single rate)
│   └── train_mas.py             # MAS recipe; evaluates on all 5 test masks
├── configs/
│   ├── hpo_results.py           # Per-model imputation HPO (paper Appendix H)
│   └── hpo_results_fcst.py      # Per-model forecasting HPO
├── run_minimal_exp.py           # Imputation smoke test (SAITS, baseline + MAS)
├── run_minimal_forecast.py      # Forecasting smoke test (TimesNet, 18→6 horizon)
├── requirement.txt
└── README.md
```

## Requirements

```bash
pip install -r requirement.txt
```

Tested with Python 3.10 + PyTorch 2.2 (CUDA 12.1). Most CPUs work for the
smoke tests; full benchmark reproduction is GPU-recommended.

> **Note on `py310pots.yml`.** This file is the conda environment export from
> our HPC cluster (Linux, aarch64, CUDA 12.1) and is included only for exact
> environment reproducibility. **Most users should ignore it** and install
> via `requirement.txt` above; the conda file pins ARM-specific package
> builds that will not resolve on x86 Linux, macOS, or Windows.

## Dataset

Hosted at https://huggingface.co/datasets/anonymous-dianchi-2026/dianchi-water.
Auto-downloaded on first call and cached at `~/.cache/dianchi_water/` — no
manual download.

```python
from mas.data_loader import load_dianchi_water
data = load_dianchi_water()
print(data["X"].shape)   # (144,496, 11)
```

## Cleaning pipeline (from raw to released parquet)

The `cleaning/` directory contains the full executable pipeline that
transforms the raw Dianchi CSV files into the released parquet. To reproduce
the released dataset from scratch:

```bash
python cleaning/pipeline.py \
  --data-dir cleaning/public_raw_dianchi \
  --years 2022 2023 2024 \
  --file-template "dianchi_raw_{year}.csv" \
  --output dianchi_data_df.parquet \
  --merge-decisions cleaning/merge_decisions.json \
  --manual-corrections cleaning/manual_corrections.json \
  --start-time "2022-01-01 00:00:00" \
  --end-time "2024-12-30 23:59:59"
```

The output `dianchi_data_df.parquet` is byte-identical to the file hosted on
HuggingFace. See `cleaning/README.md` for stage-by-stage documentation.

## Quick start

Two smoke tests verify your installation and reproduce one cell of the paper
each. Both run on CPU.

**Imputation** (SAITS @ DianchiWater, point@0.5, baseline + MAS, 5 seeds):

```bash
python run_minimal_exp.py
```

**Forecasting** (TimesNet @ DianchiWater, 18→6 horizon, 5 seeds, natural
input missingness — no synthetic injection, per paper Section 5.6):

```bash
python run_minimal_forecast.py
```

Neither script generates the cross-pattern test grid; for that, see the full
benchmark below.

## Full benchmark reproduction (imputation)

The paper's main table evaluates each model on **5 fixed test masks** so that
scores are directly comparable across models and recipes. Reproduction is a
three-step process. Examples below use **iTransformer**, one of the strongest
beneficiaries of MAS in the paper (Section 5.5); swap `--model` for any of
the eleven supported architectures (see `MODELS` in
`benchmark/train_baseline.py`).

### 1. Pre-generate the 5 baseline test sets (one-time)

```bash
python benchmark/generate_baseline.py
```

Creates `generated_datasets/dianchi_water_quality_<config>/` for the five
configurations (point@0.3/0.5/0.9, subseq@0.5, block@0.5), each containing
`train.h5`, `val.h5`, `test.h5`, `scaler.pkl`. Run once; reused by every
training script.

### 2. Reproduce the baseline (standard single-pattern recipe)

```bash
python benchmark/train_baseline.py \
    --model iTransformer \
    --dataset DianchiWater \
    --dataset_fold_path generated_datasets/dianchi_water_quality_rate05_step24_point \
    --saving_path results/baseline/iTransformer_point05 \
    --device cuda:0 \
    --n_rounds 5
```

Repeat per (model, test config) combination as needed. Per-cell numbers are
written to `metrics.json` under `--saving_path`.

### 3. Reproduce MAS

Build the MAS-augmented training set, then train:

```bash
# Build augmented train.h5 (uses the scaler from step 1 for fair comparison)
python benchmark/generate_aug.py \
    --output_dir generated_datasets_aug/dianchi_psb_rolling \
    --scaler_source_dir generated_datasets/dianchi_water_quality_rate05_step24_point \
    --patterns point subseq block --rates 0.3 0.5 0.9 \
    --stride 1 --seeds 2024

# Train; evaluates on ALL 5 baseline test sets in one run
python benchmark/train_mas.py \
    --model iTransformer \
    --dataset DianchiWater \
    --aug_dataset_dir generated_datasets_aug/dianchi_psb_rolling \
    --saving_path results/mas/iTransformer \
    --device cuda:0 \
    --n_rounds 5
```

`metrics.json` contains MAE/MSE/MRE for each of the 5 test masks (mean ± std
over 5 seeds). For iTransformer on Dianchi Water, expect MAE reductions on
the order of 15–30% across test configurations relative to the baseline.

### Cross-dataset replication (BeijingAir / ETTh1)

The same three-step recipe works for both replication datasets, which are
auto-downloaded via `tsdb` through public `benchpots`. Per-dataset defaults:

| Dataset      | `n_steps` | Point rates    | Notes                                  |
|--------------|-----------|----------------|----------------------------------------|
| DianchiWater | 24        | 0.3, 0.5, 0.9  | High natural missingness (19.8%)       |
| BeijingAir   | 24        | 0.1, 0.5, 0.9  | Low natural missingness (1.6%)         |
| ETT_h1       | 48        | 0.1, 0.5, 0.9  | No natural missingness, longer window  |

Every script accepts `--dataset DianchiWater | BeijingAir | ETT_h1`;
`n_steps`, rates, stride, etc. fall back to per-dataset defaults
automatically.

```bash
# BeijingAir with iTransformer
python benchmark/generate_baseline.py --dataset BeijingAir

python benchmark/train_baseline.py \
    --model iTransformer --dataset BeijingAir \
    --dataset_fold_path generated_datasets/beijing_air_quality_rate05_step24_point \
    --saving_path results/baseline/iTransformer_beijing_point05 \
    --device cuda:0 --n_rounds 5

python benchmark/generate_aug.py \
    --dataset BeijingAir \
    --output_dir generated_datasets_aug/beijing_psb_rolling \
    --scaler_source_dir generated_datasets/beijing_air_quality_rate01_step24_point

python benchmark/train_mas.py \
    --model iTransformer --dataset BeijingAir \
    --aug_dataset_dir generated_datasets_aug/beijing_psb_rolling \
    --saving_path results/mas/iTransformer_beijing \
    --device cuda:0 --n_rounds 5
```

For ETTh1, replace `BeijingAir` → `ETT_h1` and the dataset directory prefixes
accordingly (`beijing_air_quality_*` → `etth1_*`).

## Forecasting reproduction

`run_minimal_forecast.py` is the full forecasting entry point: looping over
the 7 architectures listed in `configs/hpo_results_fcst.py` (edit the model
name at the top of the script) and the 5 random seeds reproduces the
forecasting table in the paper's Appendix end-to-end. Each window is split
into 18 input steps and a 6-step prediction horizon; the input retains the
natural block-structured missingness without synthetic injection, making
this a substantially harder setting than fully observed forecasting
benchmarks.

```bash
python run_minimal_forecast.py
```

Output: MAE/MSE/MRE on the 6-step horizon, computed only on observed cells
of the target window (mean ± std over 5 seeds).

## MAS configuration axes

| Axis           | Parameter | Paper default                  | Effect                                             |
|----------------|-----------|--------------------------------|----------------------------------------------------|
| Pattern mixing | `patterns`| `['point','subseq','block']`   | Diverse mask structures                            |
| Rate mixing    | `rates`   | `[0.3, 0.5, 0.9]`              | Cover low–high missing regimes                     |
| Rolling stride | `stride`  | `1` (full overlap)             | `stride=8` gives 83% of gains at 3× training cost  |
| Multi-seed     | `seeds`   | `[2024]`                       | Extra mask randomness per window                   |

## Reproducibility note

Reported numbers are 5-seed means. Absolute MAE may vary by ~0.01–0.03
across hardware (different GPU, CUDA, or cuDNN versions) even with
identical seeds — this is expected non-determinism from cuDNN kernel
selection and floating-point reduction order, not a defect of the release.
The relative MAS-vs-baseline improvement is stable across the test
environments we have verified.

## License

Code: BSD-3-Clause — Dataset: CC-BY-4.0

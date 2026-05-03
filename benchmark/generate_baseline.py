"""
Generate baseline (single-pattern, single-rate) datasets.

Supports three datasets: DianchiWater, BeijingAir, ETT_h1.
- Dianchi: hand-rolled pipeline (mirrors run_minimal_exp.py), since public
  benchpots does not include preprocess_dianchi_water_quality.
- BeijingAir / ETT_h1: delegates to public benchpots.

Usage:
    python benchmark/generate_baseline.py --dataset DianchiWater
    python benchmark/generate_baseline.py --dataset BeijingAir
    python benchmark/generate_baseline.py --dataset ETT_h1
"""

import argparse
import os
import sys
import h5py
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from mas.data_loader import load_dianchi_water
from mas.mask_generator import generate_mask, compute_injection_rate
from pypots.utils.random import set_random_seed
import warnings
warnings.filterwarnings("ignore", category=Warning, module="benchpots")

# ============================================================
# Dianchi: hand-rolled pipeline
# ============================================================
def _preprocess_dianchi(rate, n_steps, pattern, **pattern_kwargs):
    df = load_dianchi_water()['X']
    feature_cols = ['TEM','PH','DO','CON','NTU','IMN','NH_N','TP','TN']
    id_col = 'section_name_unique'

    dfs = []
    for sec in df[id_col].unique():
        sec_df = df[df[id_col] == sec].copy().set_index('tm')[feature_cols]
        sec_df.columns = [f'{sec}_{c}' for c in sec_df.columns]
        dfs.append(sec_df)
    combined = pd.concat(dfs, axis=1).sort_index()

    months = combined.index.to_period('M').unique()
    train_df = combined[combined.index.to_period('M').isin(months[:22])]
    val_df   = combined[combined.index.to_period('M').isin(months[22:28])]
    test_df  = combined[combined.index.to_period('M').isin(months[28:])]

    scaler = StandardScaler()
    train_raw = scaler.fit_transform(train_df.values)
    val_raw   = scaler.transform(val_df.values)
    test_raw  = scaler.transform(test_df.values)

    def make_windows(arr):
        n = len(arr) // n_steps
        return arr[:n*n_steps].reshape(n, n_steps, -1).astype(np.float32)

    train_X_ori = make_windows(train_raw)
    val_X_ori   = make_windows(val_raw)
    test_X_ori  = make_windows(test_raw)

    if pattern == 'point':
        inject_rate = float(rate)
    else:
        inject_rate = compute_injection_rate(train_X_ori, rate, pattern, **pattern_kwargs)
    print(f'  pattern={pattern} target_rate={rate} -> inject_rate={inject_rate:.4f}')

    train_X = generate_mask(train_X_ori.copy(), inject_rate, pattern, **pattern_kwargs)
    val_X   = generate_mask(val_X_ori.copy(),   inject_rate, pattern, **pattern_kwargs)
    test_X  = generate_mask(test_X_ori.copy(),  inject_rate, pattern, **pattern_kwargs)

    return {
        "train_X": train_X, "train_X_ori": train_X_ori,
        "val_X":   val_X,   "val_X_ori":   val_X_ori,
        "test_X":  test_X,  "test_X_ori":  test_X_ori,
        "scaler":  scaler,
    }


# ============================================================
# BeijingAir / ETT: use benchpots for loading only;
#                   inject masks ourselves for calibrated rates.
# ============================================================
def _preprocess_via_benchpots(loader, n_steps, rate, pattern, **pattern_kwargs):
    """Use benchpots only for loading + scaling + windowing.
    
    Mask injection is done locally via compute_injection_rate + generate_mask
    to ensure that `rate` consistently means *target total missing rate*
    across point/subseq/block. Public benchpots' block path interprets
    `rate` differently and saturates to ~100% missing for rate >= ~0.05.
    """
    # Throwaway injection — only need the *_ori arrays
    base = loader(rate=0.1, n_steps=n_steps, pattern='point')
    train_X_ori = base['train_X_ori'].astype(np.float32)
    val_X_ori   = base['val_X_ori'].astype(np.float32)
    test_X_ori  = base['test_X_ori'].astype(np.float32)
    scaler      = base['scaler']

    if pattern == 'point':
        inject_rate = float(rate)
    else:
        inject_rate = compute_injection_rate(
            train_X_ori, rate, pattern, **pattern_kwargs
        )
    print(f'  pattern={pattern} target_rate={rate} -> inject_rate={inject_rate:.4f}')

    train_X = generate_mask(train_X_ori.copy(), inject_rate, pattern, **pattern_kwargs)
    val_X   = generate_mask(val_X_ori.copy(),   inject_rate, pattern, **pattern_kwargs)
    test_X  = generate_mask(test_X_ori.copy(),  inject_rate, pattern, **pattern_kwargs)
    
    print(f'  final missing rate: '
          f'train={np.isnan(train_X).mean():.4f}, '
          f'val={np.isnan(val_X).mean():.4f}, '
          f'test={np.isnan(test_X).mean():.4f}')

    return {
        "train_X": train_X, "train_X_ori": train_X_ori,
        "val_X":   val_X,   "val_X_ori":   val_X_ori,
        "test_X":  test_X,  "test_X_ori":  test_X_ori,
        "scaler":  scaler,
    }


def _preprocess_beijing(rate, n_steps, pattern, **pattern_kwargs):
    from benchpots.datasets import preprocess_beijing_air_quality
    return _preprocess_via_benchpots(
        preprocess_beijing_air_quality, n_steps, rate, pattern, **pattern_kwargs
    )


def _preprocess_ett(subset, rate, n_steps, pattern, **pattern_kwargs):
    from benchpots.datasets import preprocess_ett
    def loader(rate, n_steps, pattern):
        return preprocess_ett(subset=subset, rate=rate, n_steps=n_steps, pattern=pattern)
    return _preprocess_via_benchpots(loader, n_steps, rate, pattern, **pattern_kwargs)


# ============================================================
# Per-dataset config
# ============================================================
DATASET_CONFIGS = {
    "DianchiWater": dict(
        n_steps=24, tag="dianchi_water_quality",
        rates_point=[0.3, 0.5, 0.9],
        preprocess=lambda **kw: _preprocess_dianchi(**kw),
    ),
    "BeijingAir": dict(
        n_steps=24, tag="beijing_air_quality",
        rates_point=[0.1, 0.5, 0.9],
        preprocess=lambda **kw: _preprocess_beijing(**kw),
    ),
    "ETT_h1": dict(
        n_steps=48, tag="etth1",
        rates_point=[0.1, 0.5, 0.9],
        preprocess=lambda **kw: _preprocess_ett(subset='ETTh1', **kw),
    ),
}


def organize_and_save(dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        with h5py.File(os.path.join(save_dir, f"{split}.h5"), "w") as f:
            f.create_dataset("X",     data=dataset[f"{split}_X"])
            f.create_dataset("X_ori", data=dataset[f"{split}_X_ori"])
    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(dataset["scaler"], f)
    print(f"  Saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="DianchiWater",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--root", default="generated_datasets")
    args = parser.parse_args()

    set_random_seed(2024)
    cfg = DATASET_CONFIGS[args.dataset]
    n_steps, tag = cfg["n_steps"], cfg["tag"]

    configs = (
        [(r, "point", {}, f"rate{int(r*10):02d}_step{n_steps}_point")
         for r in cfg["rates_point"]] +
        [(0.5, "subseq", {"seq_len": 18},
          f"rate05_step{n_steps}_subseq_seqlen18")] +
        [(0.5, "block", {"block_len": 6, "block_width": 6},
          f"rate05_step{n_steps}_block_blocklen6")]
    )

    for rate, pattern, kwargs, suffix in configs:
        print(f"\nGenerating: {args.dataset} {pattern}@{rate} ...")
        data = cfg["preprocess"](rate=rate, n_steps=n_steps,
                                 pattern=pattern, **kwargs)
        organize_and_save(data, os.path.join(args.root, f"{tag}_{suffix}"))

    print(f"\nAll {args.dataset} baseline configs generated.")


if __name__ == "__main__":
    main()
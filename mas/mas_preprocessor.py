"""
MAS (Mask-view Augmentation Strategy) preprocessor.

Implements the four MAS axes from the paper:
1. Pattern mixing
2. Rate mixing
3. Rolling stride
4. Multi-seed mask sampling

Three dataset-specific entry points share a common MAS engine:
- preprocess_dianchi_water_quality_aug
- preprocess_beijing_air_quality_aug
- preprocess_ett_aug
"""

import os
import pickle
import numpy as np
import pandas as pd

from .mask_generator import generate_mask, compute_injection_rate, encode_combo_tag
from .data_loader import load_dianchi_water

try:
    from benchpots.utils.logging import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def _rolling_windows(arr_2d, n_steps, stride):
    """Extract rolling windows from a 2D array."""
    T, F = arr_2d.shape
    n_win = (T - n_steps) // stride + 1
    out = np.empty((n_win, n_steps, F), dtype=np.float32)
    for i in range(n_win):
        s = i * stride
        out[i] = arr_2d[s : s + n_steps]
    return out


# ============================================================
# Shared MAS engine (dataset-agnostic)
# ============================================================
def _mas_engine(
    train_X_scaled_2d,
    scaler,
    n_steps,
    patterns,
    rates,
    stride,
    seeds,
    pattern_kwargs,
    inject_rate_overrides,
    sample_size,
    task_type="imputation",
):
    """Core MAS engine: rolling windows + K mask views per window + interleaving.
    Takes a 2D scaled training array; returns the augmented bundle.
    """
    patterns = [p.lower() for p in patterns]
    assert set(patterns).issubset({"point", "subseq", "block"})

    if pattern_kwargs is None:
        pattern_kwargs = {
            "subseq": {"seq_len": 18},
            "block": {"block_len": 6, "block_width": 6},
        }
    if inject_rate_overrides is None:
        inject_rate_overrides = {}

    # ---- Rolling windows ----
    train_windows = _rolling_windows(train_X_scaled_2d, n_steps, stride)
    n_win, _, n_feat = train_windows.shape
    logger.info(f"Rolling windows: n_win={n_win}, stride={stride}")

    # ---- Resolve injection rates per (pattern, rate) ----
    resolved = {}
    for p in patterns:
        for r in rates:
            key = (p, round(float(r), 4))
            if key in inject_rate_overrides:
                resolved[key] = float(inject_rate_overrides[key])
            elif p == "point":
                resolved[key] = float(r)
            else:
                kw = pattern_kwargs.get(p, {})
                rng = np.random.RandomState(20240)
                idx = rng.choice(n_win, min(sample_size, n_win), replace=False)
                resolved[key] = compute_injection_rate(
                    train_windows[idx], r, p, **kw
                )
            logger.info(f"  {p}@{r} -> inject_rate={resolved[key]:.6f}")

    # ---- Generate K mask views per window, interleaved ----
    combos = [(p, r, s) for s in seeds for r in rates for p in patterns]
    K = len(combos)
    logger.info(f"Generating {K} mask views × {n_win} windows = {K * n_win} samples")

    masked_per_combo = []
    combo_tags = []
    for (p, r, s) in combos:
        rng_state = np.random.get_state()
        np.random.seed(int(s))
        kw = pattern_kwargs.get(p, {}) if p != "point" else {}
        inject_rate = resolved[(p, round(float(r), 4))]
        X_masked = generate_mask(train_windows, inject_rate, p, **kw)
        np.random.set_state(rng_state)
        masked_per_combo.append(X_masked.astype(np.float32))
        combo_tags.append(encode_combo_tag(p, r))

    stacked = np.stack(masked_per_combo, axis=1)  # (n_win, K, T, F)
    train_X = stacked.reshape(n_win * K, n_steps, n_feat)
    train_X_ori = train_windows.astype(np.float32)
    ori_index = np.repeat(np.arange(n_win, dtype=np.int32), K)
    combo_tag_arr = np.tile(np.array(combo_tags, dtype=np.int32), n_win)

    logger.info(f"Augmented train: X={train_X.shape}, X_ori={train_X_ori.shape}, "
                f"final missing={np.isnan(train_X).mean():.3f}")

    meta = {
        "n_steps": n_steps,
        "n_features": n_feat,
        "stride": stride,
        "patterns": patterns,
        "rates": [float(r) for r in rates],
        "seeds": [int(s) for s in seeds],
        "pattern_kwargs": pattern_kwargs,
        "n_train_windows": int(n_win),
        "n_train_samples": int(train_X.shape[0]),
        "K": K,
        "interleave": True,
        "final_train_missing_rate": float(np.isnan(train_X).mean()),
        "resolved_inject_rates": {
            f"{p}__{int(round(r*1000)):03d}": float(v)
            for (p, r), v in resolved.items()
        },
        "task_type": task_type,
    }

    return {
        "train_X": train_X,
        "train_X_ori": train_X_ori,
        "ori_index": ori_index,
        "combo_tag": combo_tag_arr,
        "scaler": scaler,
        "meta": meta,
    }


# ============================================================
# Helper: load scaled 2D training array from public benchpots
# ============================================================
def _load_baseline_train_2d_via_benchpots(loader_callable, n_steps):
    """Call a benchpots preprocess_* function with a throwaway injection
    rate; take train_X_ori (no synthetic injection, only natural missing)
    and reshape its non-overlapping windows back to a 2D series.

    Assumes benchpots produces non-overlapping windows for these datasets,
    which is the convention for imputation preprocessing.
    """
    baseline = loader_callable(rate=0.1, n_steps=n_steps, pattern='point')
    train_X_ori = baseline['train_X_ori']  # (n_win, n_steps, F)
    n_win, T, F = train_X_ori.shape
    train_X_2d = train_X_ori.reshape(n_win * T, F)
    return train_X_2d, baseline['scaler']


# ============================================================
# Per-dataset entry points
# ============================================================
def preprocess_dianchi_water_quality_aug(
    n_steps=24,
    patterns=("point", "subseq", "block"),
    rates=(0.3, 0.5, 0.9),
    stride=1,
    seeds=(2024,),
    pattern_kwargs=None,
    scaler_source_dir=None,
    inject_rate_overrides=None,
    sample_size=200,
    strict_tol=0.01,
    task_type="imputation",
    n_pred_steps=1,
    forecast_feature_indices=None,
):
    """Generate MAS-augmented training data for Dianchi Water."""
    # ---- Load + chronological split ----
    data = load_dianchi_water()
    df = data["X"]
    feature_columns = ["TEM", "PH", "DO", "CON", "NTU", "IMN", "NH_N", "TP", "TN"]
    id_col = "section_name_unique"

    df["date_time"] = pd.to_datetime(df["tm"])
    df = df.dropna(subset=["date_time"]).sort_values(by=[id_col, "date_time"])

    dfs = []
    for sec in df[id_col].unique():
        sec_df = df[df[id_col] == sec].copy().set_index("date_time")[feature_columns]
        sec_df.columns = [f"{sec}_{col}" for col in sec_df.columns]
        dfs.append(sec_df)
    combined_df = pd.concat(dfs, axis=1).sort_index()

    date_index = combined_df.index
    train_months = date_index.to_period("M").unique()[:22]
    train_df = combined_df[date_index.to_period("M").isin(train_months)]
    train_X_raw = train_df.to_numpy(dtype=float)

    logger.info(f"Dianchi train raw shape: {train_X_raw.shape}, "
                f"natural missing: {np.isnan(train_X_raw).mean():.3f}")

    # ---- Load pre-fitted scaler ----
    if scaler_source_dir is None:
        raise ValueError("scaler_source_dir is required for fair comparison")
    with open(os.path.join(scaler_source_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    train_X_scaled = scaler.transform(train_X_raw)

    return _mas_engine(
        train_X_scaled, scaler, n_steps, patterns, rates, stride, seeds,
        pattern_kwargs, inject_rate_overrides, sample_size, task_type,
    )


def preprocess_beijing_air_quality_aug(
    n_steps=24,
    patterns=("point", "subseq", "block"),
    rates=(0.1, 0.5, 0.9),
    stride=1,
    seeds=(2024,),
    pattern_kwargs=None,
    scaler_source_dir=None,
    inject_rate_overrides=None,
    sample_size=200,
    strict_tol=0.01,
    task_type="imputation",
):
    """Generate MAS-augmented training data for Beijing Multi-site Air Quality."""
    from benchpots.datasets import preprocess_beijing_air_quality
    train_X_2d, _ = _load_baseline_train_2d_via_benchpots(
        preprocess_beijing_air_quality, n_steps
    )
    logger.info(f"Beijing train_X 2D shape: {train_X_2d.shape}")

    if scaler_source_dir is None:
        raise ValueError("scaler_source_dir is required for fair comparison")
    with open(os.path.join(scaler_source_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    return _mas_engine(
        train_X_2d, scaler, n_steps, patterns, rates, stride, seeds,
        pattern_kwargs, inject_rate_overrides, sample_size, task_type,
    )


def preprocess_ett_aug(
    subset='ETTh1',
    n_steps=48,
    patterns=("point", "subseq", "block"),
    rates=(0.1, 0.5, 0.9),
    stride=1,
    seeds=(2024,),
    pattern_kwargs=None,
    scaler_source_dir=None,
    inject_rate_overrides=None,
    sample_size=200,
    strict_tol=0.01,
    task_type="imputation",
):
    """Generate MAS-augmented training data for ETT (ETTh1/h2/m1/m2)."""
    from benchpots.datasets import preprocess_ett

    def _loader(rate, n_steps, pattern):
        return preprocess_ett(subset=subset, rate=rate, n_steps=n_steps, pattern=pattern)

    train_X_2d, _ = _load_baseline_train_2d_via_benchpots(_loader, n_steps)
    logger.info(f"ETT/{subset} train_X 2D shape: {train_X_2d.shape}")

    if scaler_source_dir is None:
        raise ValueError("scaler_source_dir is required for fair comparison")
    with open(os.path.join(scaler_source_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    return _mas_engine(
        train_X_2d, scaler, n_steps, patterns, rates, stride, seeds,
        pattern_kwargs, inject_rate_overrides, sample_size, task_type,
    )
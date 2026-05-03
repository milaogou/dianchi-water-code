"""
Minimal forecasting demo: TimesNet on Dianchi Water with natural input missingness.

Pipeline:
1. Load Dianchi data (auto-download from HuggingFace)
2. Pivot to (T, 198) and chronologically split (22mo train / 7mo val / 7mo test)
3. Per-feature standardize using train statistics
4. Build 24-step windows; split each into 18 input + 6 prediction steps
   (input retains natural ~19.8% missingness; NO synthetic injection — paper Section 5.6)
5. Train TimesNet (pypots.forecasting) over 5 seeds
6. Report MAE/MSE/MRE on observed cells of the 6-step horizon

This is the forecasting counterpart of run_minimal_exp.py. Full benchmark
results for 7 forecasting models are in paper Appendix.
"""

import time
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from mas.data_loader import load_dianchi_water
from pypots.forecasting import TimesNet
from pypots.optim import Adam
from configs.hpo_results_fcst import HPO_RESULTS_FCST

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- One-off preprocessing ----------
print('=== Loading & preprocessing ===')
data = load_dianchi_water()
df = data['X']
feature_cols = ['TEM','PH','DO','CON','NTU','IMN','NH_N','TP','TN']
id_col = 'section_name_unique'

dfs = []
for sec in df[id_col].unique():
    sec_df = df[df[id_col] == sec].copy().set_index('tm')[feature_cols]
    sec_df.columns = [f'{sec}_{c}' for c in sec_df.columns]
    dfs.append(sec_df)
combined = pd.concat(dfs, axis=1).sort_index()

months   = combined.index.to_period('M').unique()
train_df = combined[combined.index.to_period('M').isin(months[:22])]
val_df   = combined[combined.index.to_period('M').isin(months[22:28])]
test_df  = combined[combined.index.to_period('M').isin(months[28:])]

scaler = StandardScaler()
train_raw = scaler.fit_transform(train_df.values)
val_raw   = scaler.transform(val_df.values)
test_raw  = scaler.transform(test_df.values)


# ---------- Forecasting windows: split 24 = 18 input + 6 predict ----------
N_STEPS = 24
N_INPUT = 18
N_PRED  = 6

def make_forecast_windows(arr):
    n = len(arr) // N_STEPS
    w = arr[:n*N_STEPS].reshape(n, N_STEPS, -1).astype(np.float32)
    return w[:, :N_INPUT], w[:, N_INPUT:]   # (X, X_pred)

train_X, train_X_pred = make_forecast_windows(train_raw)
val_X,   val_X_pred   = make_forecast_windows(val_raw)
test_X,  test_X_pred  = make_forecast_windows(test_raw)

n_features = train_X.shape[-1]
print(f'Windows: train={train_X.shape}, val={val_X.shape}, test={test_X.shape}')
print(f'Natural missing — input: {np.isnan(train_X).mean():.3f}, '
      f'target: {np.isnan(train_X_pred).mean():.3f}')


# ---------- Multi-seed loop ----------
N_ROUNDS = 5
SEED_START = 2024
log = []

for i in range(N_ROUNDS):
    seed = SEED_START + i
    print(f'\n=== Round {i+1}/{N_ROUNDS}  seed={seed} ===')
    set_seed(seed)

    cfg = HPO_RESULTS_FCST['DianchiWater']['TimesNet'].copy()
    lr  = cfg.pop('lr')
    # Pop dimension-related keys; we set them from our own pipeline to avoid
    # any mismatch with the HPO file (e.g. if HPO recorded n_steps=24).
    for k in ['n_steps', 'n_features', 'n_pred_steps', 'n_pred_features']:
        cfg.pop(k, None)
    # CPU-friendly caps for smoke test (use HPO value if it's already smaller)
    cfg['epochs']   = min(cfg.get('epochs', 100), 30)
    cfg['patience'] = min(cfg.get('patience', 10), 5)

    model = TimesNet(
        n_steps=N_INPUT, n_features=n_features,
        n_pred_steps=N_PRED, n_pred_features=n_features,
        optimizer=Adam(lr=lr),
        device='cpu',
        **cfg,
    )

    t0 = time.time()
    model.fit(
        train_set={'X': train_X, 'X_pred': train_X_pred},
        val_set  ={'X': val_X,   'X_pred': val_X_pred},
    )
    train_time = time.time() - t0

    # ---- Eval on observed cells of test_X_pred ----
    pred   = model.predict({'X': test_X})['forecasting']   # (n_test, 6, F)
    mask   = ~np.isnan(test_X_pred)
    target = np.nan_to_num(test_X_pred, nan=0.0)
    err    = (pred - target) * mask
    n_obs  = mask.sum()

    mae = np.abs(err).sum() / n_obs
    mse = (err ** 2).sum() / n_obs
    mre = np.abs(err).sum() / max(np.abs(target * mask).sum(), 1e-12)

    print(f'TimesNet forecasting on DianchiWater: '
          f'MAE={mae:.4f}, MSE={mse:.4f}, MRE={mre:.4f}, train_time={train_time:.1f}s')
    log.append((mae, mse, mre, train_time))


# ---------- Summary ----------
print('\n=== Summary over 5 rounds ===')
arr = np.array(log)
m, s = arr.mean(0), arr.std(0)
print(f'MAE        = {m[0]:.4f} ± {s[0]:.4f}')
print(f'MSE        = {m[1]:.4f} ± {s[1]:.4f}')
print(f'MRE        = {m[2]:.4f} ± {s[2]:.4f}')
print(f'train_time = {m[3]:.1f}s ± {s[3]:.1f}s')
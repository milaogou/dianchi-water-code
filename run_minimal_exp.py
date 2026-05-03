import os
import time
import random
import pickle
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from mas.data_loader import load_dianchi_water
from mas.mask_generator import generate_mask
# ↓ 路径按你项目里实际的文件名改;函数名是确定的
from mas.mas_preprocessor import preprocess_dianchi_water_quality_aug
from configs.hpo_results import HPO_RESULTS

from pypots.imputation import SAITS
from pypots.optim import Adam
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 一次性预处理:验证/测试管线 + 保存 scaler 供 MAS 复用
# ============================================================
print('=== Loading & preprocessing (val/test + scaler) ===')
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
train_raw = scaler.fit_transform(train_df.values)   # 仅用训练集 fit
val_raw   = scaler.transform(val_df.values)
test_raw  = scaler.transform(test_df.values)

# MAS 预处理器需要从磁盘加载 scaler.pkl,这里写到临时目录
scaler_dir = tempfile.mkdtemp(prefix='dianchi_scaler_')
with open(os.path.join(scaler_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

def make_windows(arr, n_steps=24):
    n = len(arr) // n_steps
    return arr[:n*n_steps].reshape(n, n_steps, -1).astype(np.float32)

train_X_ori_base = make_windows(train_raw)   # baseline 用 (stride=24)
val_X_ori        = make_windows(val_raw)
test_X_ori       = make_windows(test_raw)
print(f'Baseline windows: train={train_X_ori_base.shape}, '
      f'val={val_X_ori.shape}, test={test_X_ori.shape}')


# ============================================================
# 两种训练数据制备方案
# ============================================================
def prep_train_baseline(seed):
    """Baseline: 单 mask, point@0.5, stride=24"""
    train_X = generate_mask(train_X_ori_base.copy(), rate=0.5, pattern='point')
    return train_X, train_X_ori_base


def prep_train_mas(seed):
    """MAS full config: pat=psb, rates={0.3,0.5,0.9}, stride=1, single seed"""
    out = preprocess_dianchi_water_quality_aug(
        n_steps=24,
        patterns=('point', 'subseq', 'block'),
        rates=(0.3, 0.5, 0.9),
        stride=1,
        seeds=(seed,),
        scaler_source_dir=scaler_dir,
    )
    train_X = out['train_X']                              # (n_win*K, T, F)
    # train_X_ori 是去重存的 (n_win, T, F),用 ori_index 广播到与 train_X 同形
    train_X_ori = out['train_X_ori'][out['ori_index']]    # (n_win*K, T, F)
    print(f'  [MAS] train_X={train_X.shape}, '
          f'missing={np.isnan(train_X).mean():.3f}')
    return train_X, train_X_ori


# ============================================================
# 通用训练 + 评估
# ============================================================
def build_saits():
    cfg = HPO_RESULTS["DianchiWater"]["SAITS"].copy()
    lr  = cfg.pop("lr")
    return SAITS(**cfg, optimizer=Adam(lr=lr), device='cpu')

def run_one_round(prep_train_fn, seed):
    set_seed(seed)
    train_X, train_X_ori = prep_train_fn(seed)
    val_X  = generate_mask(val_X_ori.copy(),  rate=0.5, pattern='point')
    test_X = generate_mask(test_X_ori.copy(), rate=0.5, pattern='point')

    model = build_saits()
    t0 = time.time()
    model.fit(
        train_set={'X': train_X, 'X_ori': train_X_ori},
        val_set  ={'X': val_X,   'X_ori': val_X_ori},
    )
    train_time = time.time() - t0

    imp = model.predict({'X': test_X})['imputation']
    mask = np.isnan(test_X) & (~np.isnan(test_X_ori))
    target = np.nan_to_num(test_X_ori, nan=0.0)
    return (
        calc_mae(imp, target, mask),
        calc_mse(imp, target, mask),
        calc_mre(imp, target, mask),
        train_time,
    )

def run_experiment(label, prep_train_fn, n_rounds=5, seed_start=2024):
    print(f'\n########## {label} ##########')
    log = []
    for i in range(n_rounds):
        seed = seed_start + i
        print(f'\n--- {label}  Round {i+1}/{n_rounds}  seed={seed} ---')
        mae, mse, mre, tt = run_one_round(prep_train_fn, seed)
        print(f'{label}: MAE={mae:.4f}, MSE={mse:.4f}, '
              f'MRE={mre:.4f}, train_time={tt:.1f}s')
        log.append((mae, mse, mre, tt))
    arr = np.array(log)
    m, s = arr.mean(0), arr.std(0)
    print(f'\n=== {label}  Summary over {n_rounds} rounds ===')
    print(f'MAE        = {m[0]:.4f} ± {s[0]:.4f}')
    print(f'MSE        = {m[1]:.4f} ± {s[1]:.4f}')
    print(f'MRE        = {m[2]:.4f} ± {s[2]:.4f}')
    print(f'train_time = {m[3]:.1f}s ± {s[3]:.1f}s')
    return arr


# ============================================================
# 跑两组实验
# ============================================================
try:
    saits_log = run_experiment(
        "SAITS baseline (point@0.5, stride=24)",
        prep_train_baseline,
    )
    #Warning! MAS require huge training time……
    saits_mas_log = run_experiment(
        "SAITS + MAS (psb, rates=0.3/0.5/0.9, stride=1)",
        prep_train_mas,
    )

    # 横向对比
    print('\n========== Baseline vs MAS (mean over 5 rounds) ==========')
    for name, arr in [('Baseline', saits_log), ('MAS', saits_mas_log)]:
        m = arr.mean(0)
        print(f'{name:>8}: MAE={m[0]:.4f}  MSE={m[1]:.4f}  '
              f'MRE={m[2]:.4f}  time={m[3]:.1f}s')
    delta_mae = (saits_log[:,0].mean() - saits_mas_log[:,0].mean()) \
                / saits_log[:,0].mean() * 100
    print(f'MAS Δ MAE = {delta_mae:+.2f}%  (positive = MAS better)')
finally:
    shutil.rmtree(scaler_dir, ignore_errors=True)
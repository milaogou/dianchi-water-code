"""
Train and evaluate an imputation model under the standard single-(pattern,rate) recipe.

Usage:
    python benchmark/train_baseline.py \
        --model SAITS \
        --dataset DianchiWater \
        --dataset_fold_path generated_datasets/dianchi_water_quality_rate05_step24_point \
        --saving_path results/baseline/SAITS \
        --device cuda:0 \
        --n_rounds 5
"""

import argparse
import os
import time
import json

import numpy as np
import h5py
import torch
from pypots.imputation import *  # noqa: F403
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre
from pypots.utils.random import set_random_seed

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.hpo_results import HPO_RESULTS

RANDOM_SEEDS = [2024, 2025, 2026, 2027, 2028]

MODELS = {
    "BRITS": BRITS, "CSDI": CSDI,
    # "HELIX": HELIX,
    "iTransformer": iTransformer, "ModernTCN": ModernTCN,
    "MOMENT": MOMENT, "NonstationaryTransformer": NonstationaryTransformer,
    "PatchTST": PatchTST, "SAITS": SAITS, "StemGNN": StemGNN, "TEFN": TEFN,
}


def load_dataset(fold_path):
    """Load train/val/test from H5 files."""
    def _load(name, with_ori=False):
        path = os.path.join(fold_path, f"{name}.h5")
        with h5py.File(path, "r") as f:
            out = {"X": f["X"][:].astype(np.float32)}
            if with_ori and "X_ori" in f:
                out["X_ori"] = f["X_ori"][:].astype(np.float32)
        return out

    train = _load("train", with_ori=True)
    val = _load("val", with_ori=True)
    test = _load("test", with_ori=True)

    test_X = test["X"]
    test_X_ori = test["X_ori"]
    indicating_mask = np.isnan(test_X) & (~np.isnan(test_X_ori))
    test_X_ori = np.nan_to_num(test_X_ori, nan=0.0)

    return train, val, test_X, test_X_ori, indicating_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_fold_path", type=str, required=True)
    parser.add_argument("--saving_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_rounds", type=int, default=5)
    args = parser.parse_args()

    torch.set_num_threads(4)
    train_set, val_set, test_X, test_X_ori, test_mask = load_dataset(args.dataset_fold_path)

    collectors = {"mae": [], "mse": [], "mre": [], "time": []}

    for n_round in range(args.n_rounds):
        set_random_seed(RANDOM_SEEDS[n_round])
        round_path = os.path.join(args.saving_path, f"round_{n_round}")

        hp = HPO_RESULTS[args.dataset][args.model].copy()
        lr = hp.pop("lr")
        hp.update(device=args.device, saving_path=round_path,
                  model_saving_strategy="best", optimizer=Adam(lr=lr))

        model = MODELS[args.model](**hp)
        t0 = time.time()
        model.fit(train_set=train_set, val_set=val_set)
        train_time = time.time() - t0

        if args.model in ["CSDI", "GPVAE"]:
            imp = model.predict({"X": test_X}, n_sampling_times=10)["imputation"].mean(axis=1)
        else:
            imp = model.predict({"X": test_X})["imputation"]

        mae = float(calc_mae(imp, test_X_ori, test_mask))
        mse = float(calc_mse(imp, test_X_ori, test_mask))
        mre = float(calc_mre(imp, test_X_ori, test_mask))
        collectors["mae"].append(mae)
        collectors["mse"].append(mse)
        collectors["mre"].append(mre)
        collectors["time"].append(train_time)
        logger.info(f"Round {n_round}: MAE={mae:.4f} MSE={mse:.4f} MRE={mre:.4f}")

    summary = {
        "model": args.model, "dataset": args.dataset,
        "mae_mean": float(np.mean(collectors["mae"])),
        "mae_std": float(np.std(collectors["mae"])),
        "mse_mean": float(np.mean(collectors["mse"])),
        "mre_mean": float(np.mean(collectors["mre"])),
        "train_time_mean": float(np.mean(collectors["time"])),
    }
    os.makedirs(args.saving_path, exist_ok=True)
    with open(os.path.join(args.saving_path, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Final: MAE={summary['mae_mean']:.4f}±{summary['mae_std']:.4f}")

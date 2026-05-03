"""
Train and evaluate an imputation model under MAS augmentation.

Reads augmented train.h5 (X, X_ori, ori_index layout), trains the model,
and evaluates on ALL 5 baseline test sets in a single run.

Usage:
    python benchmark/train_mas.py \
        --model SAITS \
        --dataset DianchiWater \
        --aug_dataset_dir generated_datasets_aug/<exp_dir> \
        --saving_path results/mas/SAITS \
        --device cuda:0 \
        --n_rounds 5
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
import torch
from pypots.imputation import *  # noqa: F403
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre
from pypots.utils.random import set_random_seed

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.hpo_results import HPO_RESULTS

RANDOM_SEEDS = [2024, 2025, 2026, 2027, 2028]
BASELINE_ROOT = "generated_datasets"

MODELS = {
    "BRITS": BRITS, "CSDI": CSDI,
    # "HELIX": HELIX,
    "iTransformer": iTransformer, "ModernTCN": ModernTCN,
    "MOMENT": MOMENT, "NonstationaryTransformer": NonstationaryTransformer,
    "PatchTST": PatchTST, "SAITS": SAITS, "StemGNN": StemGNN, "TEFN": TEFN,
}

# 5 baseline test sets per dataset
DATASET_EVAL_CONFIG = {
    "DianchiWater": {
        "test_dirs": {
            "point_03": "dianchi_water_quality_rate03_step24_point",
            "point_05": "dianchi_water_quality_rate05_step24_point",
            "point_09": "dianchi_water_quality_rate09_step24_point",
            "subseq_05": "dianchi_water_quality_rate05_step24_subseq_seqlen18",
            "block_05": "dianchi_water_quality_rate05_step24_block_blocklen6",
        },
        "val_dir": "dianchi_water_quality_rate05_step24_point",
    },
    "BeijingAir": {
        "test_dirs": {
            "point_01": "beijing_air_quality_rate01_step24_point",
            "point_05": "beijing_air_quality_rate05_step24_point",
            "point_09": "beijing_air_quality_rate09_step24_point",
            "subseq_05": "beijing_air_quality_rate05_step24_subseq_seqlen18",
            "block_05": "beijing_air_quality_rate05_step24_block_blocklen6",
        },
        "val_dir": "beijing_air_quality_rate05_step24_point",
    },
    "ETT_h1": {
        "test_dirs": {
            "point_01": "etth1_rate01_step48_point",
            "point_05": "etth1_rate05_step48_point",
            "point_09": "etth1_rate09_step48_point",
            "subseq_05": "etth1_rate05_step48_subseq_seqlen18",
            "block_05": "etth1_rate05_step48_block_blocklen6",
        },
        "val_dir": "etth1_rate05_step48_point",
    },
}


def load_aug_train(train_h5_path):
    """Load MAS-augmented training data with ori_index expansion."""
    with h5py.File(train_h5_path, "r") as f:
        X = f["X"][:].astype(np.float32)
        X_ori_unique = f["X_ori"][:].astype(np.float32)
        ori_index = f["ori_index"][:]
    X_ori = X_ori_unique[ori_index]
    return {"X": X, "X_ori": X_ori}


def load_test_set(test_dir):
    """Load baseline test set and compute indicating mask."""
    with h5py.File(os.path.join(test_dir, "test.h5"), "r") as f:
        test_X = f["X"][:].astype(np.float32)
        test_X_ori = f["X_ori"][:].astype(np.float32)
    indicating_mask = np.isnan(test_X) & (~np.isnan(test_X_ori))
    test_X_ori = np.nan_to_num(test_X_ori, nan=0.0)
    return test_X, test_X_ori, indicating_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--aug_dataset_dir", type=str, required=True)
    parser.add_argument("--saving_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_rounds", type=int, default=5)
    args = parser.parse_args()

    torch.set_num_threads(4)

    # Load augmented training set
    train_set = load_aug_train(os.path.join(args.aug_dataset_dir, "train.h5"))
    logger.info(f"Aug train: X={train_set['X'].shape}, "
                f"missing={np.isnan(train_set['X']).mean():.3f}")

    # Load val + all test sets
    eval_cfg = DATASET_EVAL_CONFIG[args.dataset]
    val_h5 = os.path.join(BASELINE_ROOT, eval_cfg["val_dir"], "val.h5")
    with h5py.File(val_h5, "r") as f:
        val_set = {"X": f["X"][:].astype(np.float32),
                   "X_ori": f["X_ori"][:].astype(np.float32)}

    test_sets = {}
    for tag, dirname in eval_cfg["test_dirs"].items():
        test_sets[tag] = load_test_set(os.path.join(BASELINE_ROOT, dirname))

    # Train & evaluate
    per_round = []
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

        round_metrics = {}
        for tag, (tX, tX_ori, mask) in test_sets.items():
            if args.model in ["CSDI", "GPVAE"]:
                imp = model.predict({"X": tX}, n_sampling_times=10)["imputation"].mean(axis=1)
            else:
                imp = model.predict({"X": tX})["imputation"]
            round_metrics[tag] = {
                "mae": float(calc_mae(imp, tX_ori, mask)),
                "mse": float(calc_mse(imp, tX_ori, mask)),
                "mre": float(calc_mre(imp, tX_ori, mask)),
            }
            logger.info(f"  Round{n_round} {tag}: MAE={round_metrics[tag]['mae']:.4f}")

        per_round.append(round_metrics)

    # Aggregate & save
    summary = {"model": args.model, "dataset": args.dataset, "per_test": {}}
    for tag in eval_cfg["test_dirs"]:
        maes = [r[tag]["mae"] for r in per_round]
        summary["per_test"][tag] = {
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
        }

    os.makedirs(args.saving_path, exist_ok=True)
    with open(os.path.join(args.saving_path, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    for tag, m in summary["per_test"].items():
        logger.info(f"Final {tag}: MAE={m['mae_mean']:.4f}±{m['mae_std']:.4f}")

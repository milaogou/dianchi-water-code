"""
Generate MAS-augmented training set, written as H5 for train_mas.py.

Supports three datasets: DianchiWater, BeijingAir, ETT_h1.

Usage:
    python benchmark/generate_aug.py \
        --dataset DianchiWater \
        --output_dir generated_datasets_aug/dianchi_psb_rolling \
        --scaler_source_dir generated_datasets/dianchi_water_quality_rate05_step24_point
"""

import argparse
import json
import os
import sys
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


DATASET_DEFAULTS = {
    "DianchiWater": dict(n_steps=24, rates=[0.3, 0.5, 0.9], stride=1,
                         fn="preprocess_dianchi_water_quality_aug",
                         extra_kwargs={}),
    "BeijingAir":   dict(n_steps=24, rates=[0.1, 0.5, 0.9], stride=1,
                         fn="preprocess_beijing_air_quality_aug",
                         extra_kwargs={}),
    "ETT_h1":       dict(n_steps=48, rates=[0.1, 0.5, 0.9], stride=1,
                         fn="preprocess_ett_aug",
                         extra_kwargs={"subset": "ETTh1"}),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASET_DEFAULTS.keys()))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scaler_source_dir", required=True,
                        help="Directory containing scaler.pkl from generate_baseline.py")
    parser.add_argument("--patterns", nargs="+",
                        default=["point", "subseq", "block"])
    parser.add_argument("--rates", nargs="+", type=float, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024])
    parser.add_argument("--n_steps", type=int, default=None)
    args = parser.parse_args()

    d = DATASET_DEFAULTS[args.dataset]
    if args.rates   is None: args.rates   = d["rates"]
    if args.stride  is None: args.stride  = d["stride"]
    if args.n_steps is None: args.n_steps = d["n_steps"]

    print(f"MAS config for {args.dataset}: pat={args.patterns} rates={args.rates} "
          f"stride={args.stride} seeds={args.seeds} n_steps={args.n_steps}")

    from mas import mas_preprocessor as mp
    fn = getattr(mp, d["fn"])

    out = fn(
        n_steps=args.n_steps,
        patterns=tuple(args.patterns),
        rates=tuple(args.rates),
        stride=args.stride,
        seeds=tuple(args.seeds),
        scaler_source_dir=args.scaler_source_dir,
        **d["extra_kwargs"],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    train_h5 = os.path.join(args.output_dir, "train.h5")
    with h5py.File(train_h5, "w") as f:
        f.create_dataset("X",         data=out["train_X"])
        f.create_dataset("X_ori",     data=out["train_X_ori"])
        f.create_dataset("ori_index", data=out["ori_index"])
        f.create_dataset("combo_tag", data=out["combo_tag"])

    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(out["meta"], f, indent=2)

    print(f"\nSaved augmented train.h5 to {train_h5}")
    print(f"  X.shape={out['train_X'].shape}, "
          f"X_ori.shape={out['train_X_ori'].shape} (unique)")


if __name__ == "__main__":
    main()
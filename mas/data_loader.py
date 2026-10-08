"""
Standalone data loader for Dianchi Water dataset.

Downloads the parquet file directly from Hugging Face, eliminating the
dependency on TSDB's unreleased development branch.

Usage:
    from mas.data_loader import load_dianchi_water
    df = load_dianchi_water()  # downloads on first call, caches locally
"""

import os
import pandas as pd

HUGGINGFACE_URL = (
    "https://huggingface.co/datasets/ceepr-bit/"
    "dianchi-water/resolve/main/data/dianchi_data_df.parquet"
)
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dianchi_water")
CACHE_FILE = os.path.join(CACHE_DIR, "dianchi_data_df.parquet")

FEATURE_COLUMNS = ["TEM", "PH", "DO", "CON", "NTU", "IMN", "NH_N", "TP", "TN"]
ID_COLUMN = "station"


def _download_if_needed():
    """Download the parquet file from Hugging Face if not cached."""
    if os.path.exists(CACHE_FILE):
        # Verify file is not corrupted (minimum size check)
        if os.path.getsize(CACHE_FILE) > 100_000:
            return CACHE_FILE
        else:
            os.remove(CACHE_FILE)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Downloading Dianchi Water dataset to {CACHE_FILE} ...")

    # Use hf_hub_download but let it manage its own cache,
    # then just return the path it gives us (don't copy).
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="ceepr-bit/dianchi-water",
            filename="data/dianchi_data_df.parquet",
            repo_type="dataset",
        )
        print(f"Download complete: {path}")
        return path
    except Exception as e:
        print(f"hf_hub_download failed ({e}), trying direct URL...")

    # Fallback: direct URL download (works for public repos)
    import urllib.request
    urllib.request.urlretrieve(HUGGINGFACE_URL, CACHE_FILE)
    print(f"Download complete: {CACHE_FILE}")
    return CACHE_FILE


def load_dianchi_water(local_path=None):
    """Load and reindex the Dianchi Water dataset.

    Parameters
    ----------
    local_path : str, optional
        Path to a local copy of dianchi_data_df.parquet.
        If None, downloads from Hugging Face (cached after first call).

    Returns
    -------
    dict with key "X": pd.DataFrame
        Reindexed to a complete 4-hourly grid per station.
        Columns: tm, station, TEM, PH, DO, CON, NTU, IMN, NH_N, TP, TN
    """
    if local_path is not None:
        file_path = local_path
        if os.path.isdir(file_path):
            file_path = os.path.join(file_path, "dianchi_data_df.parquet")
    else:
        file_path = _download_if_needed()

    df = pd.read_parquet(file_path)
    df["tm"] = pd.to_datetime(df["tm"])

    # Use the column name from the HF release
    id_col = "station" if "station" in df.columns else "section_name_unique"

    df = df.sort_values([id_col, "tm"]).reset_index(drop=True)

    # Reindex each station to a complete 4-hourly grid
    full_time_index = pd.date_range(
        start=df["tm"].min(), end=df["tm"].max(), freq="4h"
    )

    dfs = []
    for station, group in df.groupby(id_col):
        group = group.set_index("tm").reindex(full_time_index)
        group[id_col] = station
        dfs.append(group.reset_index().rename(columns={"index": "tm"}))

    full_df = pd.concat(dfs, ignore_index=True)

    # Rename to the column name expected by the preprocessor
    if id_col != "section_name_unique":
        full_df = full_df.rename(columns={id_col: "section_name_unique"})

    print(f"Loaded Dianchi Water: {full_df.shape} "
          f"({full_df['section_name_unique'].nunique()} stations, "
          f"missing rate: {full_df[FEATURE_COLUMNS].isna().mean().mean():.3f})")

    return {"X": full_df}
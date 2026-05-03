import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


RE_SECTION_STATUS = re.compile(r"[\(（].*?(?:撤销|搬迁|老|新|不建).*?[\)）]")
NA_SENTINELS = ["", "--", "*", "缺失", " 缺失"]

RAW_NUMERIC_COLS = [
    "temperature",
    "pH",
    "DO",
    "conductivity",
    "NTU",
    "permanganate_index",
    "NH_N",
    "TP",
    "TN",
]

OUTPUT_COLS = ["tm", "station", "TEM", "PH", "DO", "CON", "NTU", "IMN", "NH_N", "TP", "TN"]
META_COLS = ["city", "basin", "river"]

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "public_raw_dianchi"
DEFAULT_OUTPUT = ROOT_DIR / "dianchi_data_df.parquet"
DEFAULT_MERGE_DECISIONS = SCRIPT_DIR / "merge_decisions.json"
DEFAULT_MANUAL_CORRECTIONS = SCRIPT_DIR / "manual_corrections.json"


SECTION_TO_STATION_EN = {
    "船房五社桥头(一检站)": "Chuanfang Bridge",
    "大观河入湖口": "Daguanhe Inlet",
    "断桥": "Duanqiao",
    "草海中心": "Caohai Center",
    "新河村入湖口(金属筛片厂小桥)": "Xinhecun Inlet",
    "观音山西": "Guanyinshan West",
    "王大桥": "Wangda Bridge",
    "回龙村": "Huilong Village",
    "滇池南": "Dianchi South",
    "罗家营": "Luojiaying",
    "宝丰村入湖口": "Baofengcun Inlet",
    "海口西": "Haikou West",
    "江尾下闸": "Jiangwei Lower Sluice",
    "大渔乡土罗村入湖口": "Dayuxiang Tuluocun Inlet",
    "灰湾中": "Huiwan Central",
    "白鱼口": "Baiyukou",
    "严家村桥(盘龙江)": "Yanjiancun Bridge",
    "观音山东": "Guanyinshan East",
    "观音山中": "Guanyinshan Central",
    "东大河滇池入湖口": "Dongdahe Dianchi Inlet",
    "茨巷河入湖口(牛恋乡)": "Cigang River Inlet",
    "西苑隧道": "Xiyuan Tunnel",
}


def _read_json(path: Path, default_obj):
    if not path.exists():
        return default_obj
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_section_name(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    return s.str.replace("（", "(", regex=False).str.replace("）", ")", regex=False).str.strip()


def _parse_all_dates(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y%m%d%H%M%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]
    for fmt in formats:
        mask = out.isna()
        if not mask.any():
            break
        out.loc[mask] = pd.to_datetime(s.loc[mask], format=fmt, errors="coerce")
    mask = out.isna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(s.loc[mask], errors="coerce")
    return out


def _mode_or_nan(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.mode().iloc[0]


def _load_year_csv(path: Path) -> pd.DataFrame:
    expected = ["province", "city", "basin", "river", "section_name", "tm", "water_quality_class", *RAW_NUMERIC_COLS]
    actual = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in expected if c in actual]
    df = pd.read_csv(path, dtype=str, usecols=usecols, na_values=NA_SENTINELS, keep_default_na=True)
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan
    return df[expected]


def _harmonize_station_metadata(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    group_key = ["province", "section_name"]
    for col in META_COLS:
        canonical = result.groupby(group_key, dropna=False)[col].transform(_mode_or_nan)
        result[col] = canonical.where(canonical.notna(), result[col])
    return result


def _apply_manual_corrections(df: pd.DataFrame, corrections: dict) -> pd.DataFrame:
    result = df.copy()
    mapping = corrections.get("section_name_canonicalization", {})
    if mapping:
        result["section_name"] = result["section_name"].replace(mapping)
    return result


def _fill_missing_geo_public(
    df: pd.DataFrame,
    site_info_path: str | Path | None,
    manual_corrections: dict,
    section_aliases: dict,
) -> pd.DataFrame:
    result = df.copy()

    # Step 1: deterministic in-table fill by province-level unique values.
    for col in ["basin", "river"]:
        uniq = result.groupby("province")[col].nunique(dropna=True)
        single_value_provinces = uniq[uniq == 1].index
        if len(single_value_provinces) > 0:
            prov_to_val = (
                result[result["province"].isin(single_value_provinces)]
                .groupby("province")[col]
                .agg(lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan)
            )
            mask = result[col].isna() & result["province"].isin(single_value_provinces)
            result.loc[mask, col] = result.loc[mask, "province"].map(prov_to_val)

    # Step 2: optional site info join (public-safe, local file only).
    if site_info_path is None:
        return result

    p = Path(site_info_path)
    if not p.exists():
        print(f"  site info not found, skip Stage 4 external back-fill: {p}")
        return result

    site = pd.read_csv(p, dtype=str)

    col_candidates = {
        "province": ["province", "省份"],
        "section_name": ["section_name", "断面"],
        "city": ["city", "城市"],
        "basin": ["basin", "水系"],
        "river": ["river", "河流"],
    }
    rename_map: dict[str, str] = {}
    for std_col, candidates in col_candidates.items():
        for c in candidates:
            if c in site.columns:
                rename_map[c] = std_col
                break
    site = site.rename(columns=rename_map)

    required = {"province", "section_name"}
    if not required.issubset(site.columns):
        print("  site info missing required columns (province/section_name), skip Stage 4 external back-fill")
        return result

    keep_cols = [c for c in ["province", "section_name", "city", "basin", "river"] if c in site.columns]
    site = site[keep_cols].copy()
    site["section_name"] = _normalize_section_name(site["section_name"])
    site = _apply_manual_corrections(site, manual_corrections)
    if section_aliases:
        site["section_name"] = site["section_name"].replace(section_aliases)
    site = site.drop_duplicates(subset=["province", "section_name"], keep="first")

    merged = result.merge(site, on=["province", "section_name"], how="left", suffixes=("", "_site"))
    for col in ["city", "basin", "river"]:
        site_col = f"{col}_site"
        if site_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[site_col])
            merged = merged.drop(columns=[site_col])
    return merged


def _is_decimal_precision_difference(values: np.ndarray) -> tuple[bool, float | None]:
    non_null_values = [v for v in values if not pd.isna(v)]
    if len(non_null_values) <= 1:
        return True, float(non_null_values[0]) if non_null_values else None

    try:
        numeric_values = [float(v) for v in non_null_values]
    except (ValueError, TypeError):
        return False, None

    max_val = max(numeric_values)
    min_val = min(numeric_values)

    if max_val == min_val:
        return True, float(max_val)

    abs_diff = max_val - min_val
    if abs_diff < 1e-6:
        return True, float(round(sum(numeric_values) / len(numeric_values), 6))

    rel_diff = abs_diff / max(abs(max_val), abs(min_val), 1e-10)
    if rel_diff < 0.001:
        return True, float(round(sum(numeric_values) / len(numeric_values), 6))

    for decimals in range(0, 6):
        rounded = [round(v, decimals) for v in numeric_values]
        if len(set(rounded)) == 1:
            return True, float(rounded[0])

    return False, None


def _reconcile_duplicates_by_station_time(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["section_name_unique", "tm"]
    dup_mask = df.duplicated(key_cols, keep=False)
    if not dup_mask.any():
        print("Stage 7 duplicate reconciliation: no duplicate groups")
        return df

    value_cols = ["TEM", "PH", "DO", "CON", "NTU", "IMN", "NH_N", "TP", "TN"]
    unique_df = df[~dup_mask].copy()
    dup_df = df[dup_mask].copy()

    exact_groups = 0
    precision_groups = 0
    mean_groups = 0

    merged_rows = []
    for (section_name_unique, tm), g in dup_df.groupby(key_cols, sort=False):
        g = g.sort_index()
        out = {"section_name_unique": section_name_unique, "tm": tm, "station": g["station"].iloc[0]}
        group_exact = True
        group_precision = True

        for c in value_cols:
            vals = g[c].dropna().to_numpy()
            if len(vals) == 0:
                out[c] = np.nan
                continue
            if len(vals) == 1:
                out[c] = float(vals[0])
                continue

            diff = float(np.nanmax(vals) - np.nanmin(vals))
            if diff == 0:
                out[c] = float(vals[0])
                continue

            group_exact = False
            is_decimal_diff, unified_value = _is_decimal_precision_difference(vals)
            if is_decimal_diff:
                out[c] = float(unified_value) if unified_value is not None else np.nan
                continue

            group_precision = False
            out[c] = float(np.nanmean(vals))

        merged_rows.append(out)
        if group_exact:
            exact_groups += 1
        elif group_precision:
            precision_groups += 1
        else:
            mean_groups += 1

    merged_df = pd.DataFrame(merged_rows, columns=["section_name_unique", "tm", "station", *value_cols])
    result = pd.concat([unique_df, merged_df], ignore_index=True)
    result = result.sort_values(["station", "tm"]).reset_index(drop=True)
    print(
        "Stage 7 duplicate reconciliation:",
        f"exact={exact_groups}, precision={precision_groups}, mean={mean_groups}",
    )
    return result


def build_dianchi_parquet(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    years: tuple[int, ...] = (2022, 2023, 2024),
    file_template: str = "dianchi_raw_{year}.csv",
    input_files: list[str] | None = None,
    input_is_dianchi: bool = True,
    output_path: str | Path = DEFAULT_OUTPUT,
    merge_decisions_path: str | Path = DEFAULT_MERGE_DECISIONS,
    manual_corrections_path: str | Path = DEFAULT_MANUAL_CORRECTIONS,
    site_info_path: str | Path | None = None,
    start_time: str | None = "2022-01-01 08:00:00",
    end_time: str | None = "2024-12-30 20:00:00",
) -> pd.DataFrame:
    merge_decisions = _read_json(Path(merge_decisions_path), default_obj={})
    manual_corrections = _read_json(Path(manual_corrections_path), default_obj={})

    print("Stage 1: ingestion and schema normalization")
    data_dir = Path(data_dir)
    frames = []
    paths = []
    if input_files:
        paths = [Path(p) if Path(p).is_absolute() else (data_dir / p) for p in input_files]
    else:
        for y in years:
            paths.append(data_dir / file_template.format(year=y))

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(str(path))
        frames.append(_load_year_csv(path))

    df = pd.concat(frames, ignore_index=True)
    df["tm"] = _parse_all_dates(df["tm"])
    df = df[df["tm"].notna()].copy()
    # Apply released time window early so final Stage 8 remains row-preserving.
    if start_time is not None:
        df = df[df["tm"] >= pd.Timestamp(start_time)].copy()
    if end_time is not None:
        df = df[df["tm"] <= pd.Timestamp(end_time)].copy()
    print(f"  rows after ingestion: {len(df)}")

    print("Stage 2: station status filtering")
    df["section_name"] = _normalize_section_name(df["section_name"])
    # Apply manual canonicalization before status filtering so explicitly
    # merged legacy names (e.g., xxx(老)) are retained as canonical stations.
    df["section_name"] = _apply_manual_corrections(df, manual_corrections)["section_name"]
    before = len(df)
    df = df[~df["section_name"].str.contains(RE_SECTION_STATUS, na=False)].copy()
    print(f"  removed by status markers: {before - len(df)}")

    print("Stage 3: metadata harmonization")
    df = _harmonize_station_metadata(df)

    print("Stage 4: city/watershed back-fill")
    section_aliases = merge_decisions.get("section_aliases", {})
    _ = merge_decisions.get("blacklist_pairs", [])
    df = _fill_missing_geo_public(
        df=df,
        site_info_path=site_info_path,
        manual_corrections=manual_corrections,
        section_aliases=section_aliases,
    )

    print("Stage 5: station identity merging and Dianchi extraction")
    df["section_name"] = _apply_manual_corrections(df, manual_corrections)["section_name"]
    section_aliases = merge_decisions.get("section_aliases", {})
    if section_aliases:
        df["section_name"] = df["section_name"].replace(section_aliases)

    if input_is_dianchi:
        df = df[df["province"] == "云南省"].copy()
    else:
        in_dianchi_basin = df["basin"].eq("滇池流域")
        in_known_names = df["section_name"].isin(SECTION_TO_STATION_EN.keys())
        df = df[(df["province"] == "云南省") & (in_dianchi_basin | in_known_names)].copy()

    df["station"] = df["section_name"].map(SECTION_TO_STATION_EN)
    df = df[df["station"].notna()].copy()
    print(f"  mapped stations: {df['station'].nunique()}, rows: {len(df)}")

    print("Stage 6: out-of-range remapping")
    class_mapping = {"Ⅰ": 1, "Ⅱ": 2, "Ⅲ": 3, "Ⅳ": 4, "Ⅴ": 5, "劣Ⅴ": 6, "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "劣V": 6}
    wc = df["water_quality_class"].astype(str).replace({"-3": np.nan, "-2": np.nan, "-1": np.nan})
    df["water_quality_class"] = wc.map(class_mapping).where(wc.notna(), np.nan)
    for c in RAW_NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] < 0, c] = np.nan

    print("Stage 7: duplicate reconciliation")
    df = df.rename(
        columns={
            "temperature": "TEM",
            "pH": "PH",
            "conductivity": "CON",
            "permanganate_index": "IMN",
        }
    )
    df["section_name_unique"] = df["province"].astype(str) + "_" + df["section_name"].astype(str)
    df = df[["section_name_unique", *OUTPUT_COLS]].copy()
    df = df.sort_values(["station", "tm"]).reset_index(drop=True)
    df = _reconcile_duplicates_by_station_time(df)
    df = df[OUTPUT_COLS].copy()

    print("Stage 8: final manual corrections and output")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  saved: {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument(
        "--file-template",
        type=str,
        default="dianchi_raw_{year}.csv",
        help="Input filename template used with --years, e.g. 'dianchi_raw_{year}.csv' or '{year}.csv'.",
    )
    parser.add_argument(
        "--input-files",
        type=str,
        nargs="+",
        default=None,
        help="Explicit input files (absolute paths or relative to --data-dir). Overrides --years and --file-template.",
    )
    parser.add_argument("--merge-decisions", type=str, default=str(DEFAULT_MERGE_DECISIONS))
    parser.add_argument("--manual-corrections", type=str, default=str(DEFAULT_MANUAL_CORRECTIONS))
    parser.add_argument(
        "--site-info",
        type=str,
        default=None,
        help="Optional local CSV used in Stage 4 to fill missing city/basin/river via province+section_name join.",
    )
    parser.add_argument(
        "--input-is-dianchi",
        action="store_true",
        default=True,
        help="Treat input files as already extracted Dianchi raw data (default: true).",
    )
    parser.add_argument(
        "--input-is-national",
        dest="input_is_dianchi",
        action="store_false",
        help="Treat input files as national/raw wide table and perform Dianchi extraction in Stage 5.",
    )
    parser.add_argument("--start-time", type=str, default="2022-01-01 08:00:00")
    parser.add_argument("--end-time", type=str, default="2024-12-30 20:00:00")
    args = parser.parse_args()

    df = build_dianchi_parquet(
        data_dir=args.data_dir,
        years=tuple(args.years),
        file_template=args.file_template,
        input_files=args.input_files,
        input_is_dianchi=args.input_is_dianchi,
        output_path=args.output,
        merge_decisions_path=args.merge_decisions,
        manual_corrections_path=args.manual_corrections,
        site_info_path=args.site_info,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    print(df.shape)
    print("stations", df["station"].nunique())
    print("tm", df["tm"].min(), df["tm"].max())


if __name__ == "__main__":
    main()

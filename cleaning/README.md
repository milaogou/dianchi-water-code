# Dianchi Cleaning Pipeline

This directory contains the executable cleaning pipeline for the public Dianchi subset.

## Files

```text
cleaning/
├── pipeline.py
├── merge_decisions.json
├── manual_corrections.json
├── README.md
└── public_raw_dianchi/
    ├── dianchi_raw_2022.csv
    ├── dianchi_raw_2023.csv
    └── dianchi_raw_2024.csv
```

## Scope

The cleaning pipeline was developed for and validated on the 22 Dianchi Lake basin stations, whose manageable count permits exhaustive manual auditing of merge decisions and name canonicalization.
The full national pipeline logic is eight-stage and general-purpose, but this release only includes the Dianchi subset and public-safe artifacts.

The released implementation addresses four quality issue classes:

- station identity conflicts from cross-province name collisions, handled via curated alias rules and merge-decision hooks
- missing geographic metadata, handled by in-table harmonization for this public release
- upstream sentinel codes (`-1`, `-2`, `-3`) and other negative values remapped to `NaN`
- duplicate `(station, timestamp)` rows reconciled by precision-aware rules and uncertainty-threshold averaging

The released Dianchi subset contains 116,783 cleaned records across 22 stations.

## Stage Summary

`pipeline.py` implements an eight-stage workflow:

1. Ingestion and schema normalization
2. Station status filtering
3. Metadata harmonization
4. City/watershed back-fill (public-safe subset logic)
5. Station identity merging and Dianchi extraction
6. Out-of-range remapping
7. Duplicate reconciliation
8. Final manual corrections and output

## Run From Public Raw Dianchi Files

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
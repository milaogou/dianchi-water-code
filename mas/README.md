# MAS: Mask-view Augmentation Strategy

Core implementation of the MAS protocol described in the paper. This README
documents the code organization and non-obvious design decisions; for the
conceptual definition of MAS, see paper Section 4.

## Files

- `mask_generator.py` — Three mask pattern generators (point, subseq, block)
  plus injection-rate calibration via binary search.
- `mas_preprocessor.py` — The MAS data-loading pipeline shared across the
  three datasets: rolling window extraction, Cartesian product over
  (pattern × rate × seed), interleaved assembly, and `ori_index`-based
  storage optimization.

## Key implementation details

These are choices that matter for correctness or compute and aren't obvious
from reading the source alone:

1. **Scaler sharing.** The `StandardScaler` is fit once on the baseline
   (non-augmented) training data and reused across every MAS configuration.
   This guarantees that gains under MAS come from mask diversity, not from
   shifted normalization statistics.

2. **Interleaving.** Mask views from different `(pattern, rate)` combos are
   interleaved so each minibatch sees a uniform mix, not a homogeneous block
   of one configuration. Without this, the model effectively trains on each
   `(pattern, rate)` sequentially and loses most of the diversity benefit.

3. **`ori_index` storage.** Ground-truth windows are stored once per unique
   window; each augmented sample stores only an index pointer back to its
   source. Storage cost scales as O(n_windows + K · n_windows · pointer_size)
   instead of O(K · n_windows · window_size), saving ~K-fold disk I/O.

4. **Injection-rate calibration.** For point masks, the parameter passed to
   the generator equals the target missing rate exactly. For subseq/block,
   the relationship is nonlinear, so `compute_injection_rate` binary-searches
   for the parameter that achieves the target *total* missing rate on a
   representative subsample.
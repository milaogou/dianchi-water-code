"""
Mask generation utilities for MAS (Mask-view Augmentation Strategy).

Provides three mask patterns used in the paper:
- point: MCAR (Missing Completely At Random) - independent per-cell masking
- subseq: subsequence gaps along the time axis for a single variable
- block: cross-variable block gaps (all variables masked simultaneously)

These are wrappers around BenchPOTS/PyPOTS `create_missingness`, documented
here for clarity and standalone reproducibility.
"""

import numpy as np


def generate_mask(X, rate, pattern, **kwargs):
    """Generate a missingness mask on observed entries of X.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_steps, n_features)
        Input data. NaN entries are treated as naturally missing and
        are never additionally masked.
    rate : float
        Target injection rate applied to *observed* entries.
        The final missing rate will be: natural_rate + rate * (1 - natural_rate).
    pattern : str
        One of 'point', 'subseq', 'block'.
    **kwargs :
        Pattern-specific parameters:
        - subseq: `seq_len` (int) - length of each subsequence gap. Default 18.
        - block: `block_len` (int) - temporal length of block. Default 6.
                 `block_width` (int) - number of variables per block. Default 6.

    Returns
    -------
    X_masked : np.ndarray
        Copy of X with additional NaN entries injected.
    """
    from benchpots.utils.missingness import create_missingness
    return create_missingness(X.copy(), rate, pattern, **kwargs)


def compute_injection_rate(X, target_total_rate, pattern, **kwargs):
    """Binary-search for the injection rate that achieves a target total missing rate.

    For point masks, injection_rate == target_rate (analytically exact).
    For subseq/block, the relationship is nonlinear, so we search.

    Parameters
    ----------
    X : np.ndarray
        Representative data sample for calibration.
    target_total_rate : float
        Desired total missing rate (natural + injected).
    pattern : str
        One of 'point', 'subseq', 'block'.

    Returns
    -------
    injection_rate : float
        The rate to pass to `generate_mask` to achieve the target.
    """
    if pattern == "point":
        return float(target_total_rate)

    best_rate, best_diff = None, float("inf")
    low, high = 1e-5, 0.999

    for _ in range(25):
        mid = (low + high) / 2.0
        X_masked = generate_mask(X, mid, pattern, **kwargs)
        cur = float(np.isnan(X_masked).mean())
        diff = abs(cur - target_total_rate)
        if diff < best_diff:
            best_diff = diff
            best_rate = mid
        if diff <= 1e-3:
            return mid
        if cur < target_total_rate:
            low = mid
        else:
            high = mid
    return best_rate


# ---------------------------------------------------------------------------
# Combo tag encoding (used for dataset filtering / interleaving)
# ---------------------------------------------------------------------------
PATTERN_CODE = {"point": 0, "subseq": 1, "block": 2}

def encode_combo_tag(pattern, rate):
    """Encode (pattern, rate) into an integer tag.
    
    Examples: (point, 0.3) -> 300, (subseq, 0.5) -> 10500, (block, 0.5) -> 20500
    """
    return PATTERN_CODE[pattern] * 10000 + int(round(rate * 1000))

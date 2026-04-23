"""Graph inference atoms for neural circuit reconstruction from calcium imaging.

Pure functions that implement the core algorithmic primitives from the
1st-place Kaggle Connectomics solution (Sutera et al. 2014). These atoms
are composable building blocks for inferring pairwise connectivity from
multivariate time series, applicable beyond calcium imaging to any
"infer graph from correlated signals" problem.

Source: Sutera et al., "Simple connectome inference from partial
correlation statistics in calcium imaging", arXiv:1406.7865.
License: BSD 3-Clause (original code by AAAGV team).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_calcium_lowpass_filter,
    witness_fluorescence_hard_threshold,
    witness_global_activity_sample_reweighting,
    witness_score_matrix_normalization,
    witness_temporal_precedence_directivity,
)


# ---------------------------------------------------------------------------
# Atom 1: Calcium lowpass filter
# ---------------------------------------------------------------------------


@register_atom(witness_calcium_lowpass_filter)
@icontract.require(lambda X: X.ndim == 2 and X.shape[0] >= 3, "X must be 2D with at least 3 timesteps")
@icontract.require(
    lambda kernel: kernel in {"symmetric_3", "causal_4", "forward_3", "forward_4"},
    "kernel must be one of symmetric_3, causal_4, forward_3, forward_4",
)
@icontract.ensure(lambda result, X: result.shape == X.shape, "output shape must match input")
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))), "output must be finite")
def calcium_lowpass_filter(
    X: NDArray[np.float64],
    kernel: str = "symmetric_3",
) -> NDArray[np.float64]:
    """Apply a causal or acausal weighted moving average to a time series matrix.

    Uses np.roll (circular shift) along the time axis to avoid edge artifacts
    in downstream correlation computation. Multiple kernel variants produce
    different smoothing profiles that are ensembled later for implicit
    regularization.

    The calcium indicator is itself a low-pass filter of underlying spike
    trains; this additional smoothing suppresses single-frame noise without
    requiring a full deconvolution model.

    Args:
        X: Fluorescence time series, shape (T, N). T timesteps, N neurons.
        kernel: Smoothing kernel variant:
            - "symmetric_3": [1, 1, 1] centered (t-1, t, t+1)
            - "causal_4": [1, 1, 0.8, 0.4] causal (t, t-1, t-2, t-3)
            - "forward_3": [1, 1, 1, 1] forward-looking (t, t+1, t+2, t-1)
            - "forward_4": [1, 1, 1, 1] forward-looking (t, t+1, t+2, t+3)

    Returns:
        Smoothed time series, same shape as X.
    """
    if kernel == "symmetric_3":
        return np.asarray(X + np.roll(X, -1, axis=0) + np.roll(X, 1, axis=0), dtype=np.float64)
    if kernel == "causal_4":
        return np.asarray(
            X + np.roll(X, 1, axis=0) + 0.8 * np.roll(X, 2, axis=0) + 0.4 * np.roll(X, 3, axis=0),
            dtype=np.float64,
        )
    if kernel == "forward_3":
        return np.asarray(
            X + np.roll(X, -1, axis=0) + np.roll(X, -2, axis=0) + np.roll(X, 1, axis=0),
            dtype=np.float64,
        )
    # forward_4
    return np.asarray(
        X + np.roll(X, -1, axis=0) + np.roll(X, -2, axis=0) + np.roll(X, -3, axis=0),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Atom 2: Fluorescence hard threshold
# ---------------------------------------------------------------------------


@register_atom(witness_fluorescence_hard_threshold)
@icontract.require(lambda X: X.ndim == 2, "X must be 2D")
@icontract.require(lambda threshold: threshold >= 0.0, "threshold must be non-negative")
@icontract.ensure(lambda result, X: result.shape == X.shape, "output shape must match input")
@icontract.ensure(lambda result: bool(np.all(result >= 0.0)), "output must be non-negative")
def fluorescence_hard_threshold(
    X: NDArray[np.float64],
    threshold: float = 0.11,
) -> NDArray[np.float64]:
    """Zero all values below a threshold, keeping values above unchanged.

    Applied after differencing the smoothed fluorescence signal, this
    operates on the derivative — zeroing sub-threshold noise to focus
    the subsequent correlation estimation exclusively on co-activation
    events (putative spike rising edges).

    The threshold is typically swept over a dense grid (e.g., 0.100 to
    0.210 in 120 steps) and all results are ensembled rather than
    selecting a single best value.

    Args:
        X: Derivative of smoothed fluorescence, shape (T, N).
        threshold: Values below this are zeroed. Default 0.11 from
            the original solution.

    Returns:
        Thresholded signal, same shape. Non-negative.
    """
    result = np.where(X >= threshold, X, 0.0)
    return np.asarray(result, dtype=np.float64)


# ---------------------------------------------------------------------------
# Atom 3: Global activity sample reweighting
# ---------------------------------------------------------------------------


@register_atom(witness_global_activity_sample_reweighting)
@icontract.require(lambda X: X.ndim == 2 and X.shape[0] >= 1, "X must be 2D with at least 1 timestep")
@icontract.require(lambda X: bool(np.all(X >= 0.0)), "X must be non-negative (post-threshold)")
@icontract.ensure(lambda result, X: result.shape == X.shape, "output shape must match input")
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))), "output must be finite")
def global_activity_sample_reweighting(
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Nonlinear per-timestep reweighting that suppresses global burst periods.

    For each timestep i, computes Sum4[i] = total activity across all neurons,
    then transforms: X[i,:] = (X[i,:] + 1)^(1 + 1/Sum4[i]). During high
    global burst periods (large Sum4), the exponent is near 1 (identity
    transform). During moderate activity (small Sum4), the exponent is large,
    amplifying the signal.

    This effectively deconfounds global network bursts from pairwise
    connectivity: time steps where many neurons fire simultaneously due to
    cascades are suppressed, while time steps with selective co-activation
    are amplified.

    The technique generalizes beyond calcium imaging to any multivariate
    time series where global events create spurious pairwise correlations.

    Args:
        X: Non-negative activation matrix, shape (T, N). Typically
            the output of fluorescence_hard_threshold.

    Returns:
        Reweighted activation matrix, same shape.
    """
    result = X.copy()
    global_activity = np.sum(result, axis=1)

    for i in range(result.shape[0]):
        if global_activity[i] > 0.0:
            exponent = 1.0 + 1.0 / global_activity[i]
            result[i, :] = (result[i, :] + 1.0) ** exponent
        else:
            result[i, :] = 1.0

    return np.asarray(result, dtype=np.float64)


# ---------------------------------------------------------------------------
# Atom 4: Temporal precedence directivity
# ---------------------------------------------------------------------------


@register_atom(witness_temporal_precedence_directivity)
@icontract.require(lambda X: X.ndim == 2 and X.shape[0] >= 2, "X must be 2D with at least 2 timesteps")
@icontract.require(lambda X: X.shape[1] >= 2, "X must have at least 2 neurons")
@icontract.require(lambda low: low >= 0.0, "low bound must be non-negative")
@icontract.require(lambda low, high: high > low, "high must exceed low")
@icontract.ensure(
    lambda result: result.ndim == 2 and result.shape[0] == result.shape[1],
    "output must be a square matrix",
)
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))), "output must be finite")
def temporal_precedence_directivity(
    X: NDArray[np.float64],
    low: float = 0.2,
    high: float = 0.5,
) -> NDArray[np.float64]:
    """Compute asymmetric directional connectivity via temporal precedence.

    For each ordered pair (i, j), counts time steps where x_j[t+1] falls
    in the interval [x_i[t] + low, x_i[t] + high]. Builds an asymmetric
    count matrix and returns count - count^T, capturing the directional
    signature: if neuron i fires at time t, downstream neuron j should
    show a fluorescence increase at t+1 in a specific range.

    The default interval [+0.2, +0.5] is tuned to calcium indicator
    biophysics. This atom should typically be weighted very lightly
    (e.g., 0.3%) as a directional tiebreaker on top of a symmetric
    precision matrix estimate.

    Args:
        X: Preprocessed fluorescence, shape (T, N). Should be the
            output of the lowpass → diff → threshold → reweight pipeline.
        low: Lower bound of the temporal response interval (default 0.2).
        high: Upper bound of the temporal response interval (default 0.5).

    Returns:
        Asymmetric directivity score matrix, shape (N, N).
    """
    n_neurons = X.shape[1]
    count = np.zeros((n_neurons, n_neurons), dtype=np.float64)

    x_prev = X[:-1, :]  # (T-1, N)
    x_next = X[1:, :]   # (T-1, N)

    for j in range(n_neurons):
        lower_bound = x_prev[:, j] + low   # (T-1,)
        upper_bound = x_prev[:, j] + high   # (T-1,)
        for k in range(n_neurons):
            if k == j:
                continue
            count[j, k] = float(np.sum(
                (x_next[:, k] > lower_bound) & (x_next[:, k] < upper_bound)
            ))

    return np.asarray(count - count.T, dtype=np.float64)


# ---------------------------------------------------------------------------
# Atom 5: Score matrix normalization
# ---------------------------------------------------------------------------


@register_atom(witness_score_matrix_normalization)
@icontract.require(
    lambda X: X.ndim == 2 and X.shape[0] == X.shape[1],
    "X must be a square matrix",
)
@icontract.require(lambda X: X.shape[0] >= 2, "X must have at least 2 nodes")
@icontract.ensure(lambda result: bool(np.all(result >= 0.0)) and bool(np.all(result <= 1.0)), "output must be in [0, 1]")
@icontract.ensure(
    lambda result: result.ndim == 2 and result.shape[0] == result.shape[1],
    "output must be a square matrix",
)
def score_matrix_normalization(
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize a score matrix to [0, 1] with diagonal set to minimum.

    Sets diagonal to the matrix minimum (removing self-connections), then
    applies min-max normalization. This is the final step before producing
    a connectivity score suitable for evaluation.

    Args:
        X: Raw score matrix, shape (N, N).

    Returns:
        Normalized score matrix in [0, 1], same shape. Diagonal values
        are the minimum.
    """
    result = X.copy()
    np.fill_diagonal(result, result.min())
    flat = result.ravel()
    vmin = flat.min()
    vmax = flat.max()
    if vmax - vmin < 1e-15:
        return np.zeros_like(result)
    flat = (flat - vmin) / (vmax - vmin)
    return np.asarray(flat.reshape(result.shape), dtype=np.float64)

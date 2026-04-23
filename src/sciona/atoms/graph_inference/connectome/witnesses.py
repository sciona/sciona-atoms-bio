"""Ghost witnesses for graph inference connectome atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_calcium_lowpass_filter(
    X: AbstractArray,
    kernel: str = "symmetric_3",
) -> AbstractArray:
    """Smoothed time series, same shape as input."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_fluorescence_hard_threshold(
    X: AbstractArray,
    threshold: float = 0.11,
) -> AbstractArray:
    """Thresholded signal, same shape as input."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_global_activity_sample_reweighting(
    X: AbstractArray,
) -> AbstractArray:
    """Reweighted activation matrix, same shape as input."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_temporal_precedence_directivity(
    X: AbstractArray,
    low: float = 0.2,
    high: float = 0.5,
) -> AbstractArray:
    """Asymmetric directivity score matrix, shape (N, N)."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    n = X.shape[1]
    return AbstractArray(shape=(n, n), dtype="float64")


def witness_score_matrix_normalization(
    X: AbstractArray,
) -> AbstractArray:
    """Normalized score matrix in [0, 1], same shape."""
    if len(X.shape) != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square matrix")
    return AbstractArray(shape=X.shape, dtype="float64")

from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_fill_scoring_matrix(seq1: AbstractScalar | str, seq2: AbstractScalar | str, match_score: AbstractScalar | float, mismatch_score: AbstractScalar | float, gap_penalty: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for fill_scoring_matrix."""
    _ = (seq1, seq2, match_score, mismatch_score, gap_penalty)
    return AbstractArray(shape=(), dtype="float64")

def witness_execute_traceback(seq1: AbstractScalar | str, seq2: AbstractScalar | str, scoring_matrix: AbstractArray) -> AbstractScalar:
    """Ghost witness for execute_traceback."""
    _ = (seq1, seq2, scoring_matrix)
    return AbstractScalar(dtype="float64")


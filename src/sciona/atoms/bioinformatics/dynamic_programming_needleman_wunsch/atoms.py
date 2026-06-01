from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_fill_scoring_matrix,
    witness_execute_traceback,
)

@register_atom(witness_fill_scoring_matrix, name="fill_scoring_matrix")
@icontract.require(lambda seq1, seq2, match_score, mismatch_score, gap_penalty: len(seq1) >= 0, "Precondition failed: len(seq1) >= 0")
@icontract.require(lambda seq1, seq2, match_score, mismatch_score, gap_penalty: len(seq2) >= 0, "Precondition failed: len(seq2) >= 0")
@icontract.ensure(lambda result, seq1, seq2, match_score, mismatch_score, gap_penalty: result is not None, "Postcondition failed: result is not None")
def fill_scoring_matrix(seq1: str, seq2: str, match_score: float, mismatch_score: float, gap_penalty: float) -> NDArray[np.float64]:
    """Fill DP matrix based on match, mismatch and gap costs.

    Args:
        seq1: str
        seq2: str
        match_score: float
        mismatch_score: float
        gap_penalty: float

    Returns:
        scoring_matrix: NDArray[np.float64]
    """
    return needs_human_decision() # type: ignore

@register_atom(witness_execute_traceback, name="execute_traceback")
@icontract.require(lambda seq1, seq2, scoring_matrix: seq1 is not None, "Precondition failed: seq1 is not None")
@icontract.ensure(lambda result, seq1, seq2, scoring_matrix: result is not None, "Postcondition failed: result is not None")
def execute_traceback(seq1: str, seq2: str, scoring_matrix: NDArray[np.float64]) -> str:
    """Walk back through scoring matrix grid to generate optimal alignments.

    Args:
        seq1: str
        seq2: str
        scoring_matrix: NDArray[np.float64]

    Returns:
        aligned_seq1: str
    """
    return needs_human_decision() # type: ignore


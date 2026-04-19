from __future__ import annotations

from typing import Sequence
"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np

import icontract
from sciona.ghost.registry import register_atom
from .encoding_dist_mat_witnesses import witness_encodedistancematrix

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_encodedistancematrix)
@icontract.require(lambda mat_list: mat_list is not None, "mat_list cannot be None")
@icontract.require(lambda max_cdr3: max_cdr3 > 0, "max_cdr3 must be positive")
@icontract.require(lambda max_epi: max_epi > 0, "max_epi must be positive")
@icontract.ensure(lambda result: result is not None, "EncodeDistanceMatrix output must not be None")
def encodedistancematrix(
    mat_list: Sequence[np.ndarray],
    max_cdr3: int = 20,
    max_epi: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode TCR CDR3/epitope distance matrices and their valid-value mask.

    Args:
        mat_list: Distance matrices shaped ``(len_cdr3, len_epi)``.
        max_cdr3: Maximum encoded CDR3 axis length.
        max_epi: Maximum encoded epitope axis length.

    Returns:
        ``(encoding, masking)`` arrays shaped ``(n, max_cdr3, max_epi)``.
    """
    encoding = np.zeros((len(mat_list), max_cdr3, max_epi), dtype="float32")
    masking = np.zeros((len(mat_list), max_cdr3, max_epi), dtype=bool)

    for i, mat in enumerate(mat_list):
        len_cdr3, len_epi = mat.shape
        if len_cdr3 > max_cdr3 or len_epi > max_epi:
            raise ValueError(
                f"matrix {i} shape {mat.shape} exceeds ({max_cdr3}, {max_epi})"
            )

        i_start_cdr3 = max_cdr3 // 2 - len_cdr3 // 2
        if len_epi == 8:
            i_start_epi = 2
        elif len_epi in (9, 10):
            i_start_epi = 1
        else:
            i_start_epi = 0

        cdr3_slice = slice(i_start_cdr3, i_start_cdr3 + len_cdr3)
        epi_slice = slice(i_start_epi, i_start_epi + len_epi)
        encoding[i, cdr3_slice, epi_slice] = mat
        masking[i, cdr3_slice, epi_slice] = True

    return encoding, masking

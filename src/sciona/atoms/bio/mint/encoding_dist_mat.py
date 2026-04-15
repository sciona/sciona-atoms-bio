from __future__ import annotations

from typing import List, Any, Optional, Tuple, Dict
"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np

import icontract
from sciona.ghost.registry import register_atom
from .encoding_dist_mat_witnesses import witness_encodedistancematrix

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_encodedistancematrix)
@icontract.require(lambda mat_list: isinstance(mat_list, np.ndarray), "mat_list must be a numpy array")
@icontract.ensure(lambda result: result is not None, "EncodeDistanceMatrix output must not be None")
def encodedistancematrix(mat_list: List[np.ndarray], max_cdr3: int, max_epi: int) -> np.ndarray:
    """Takes a list of matrices and pads them to a specified maximum dimension, effectively creating a batched and padded distance matrix representation.

    Args:
        mat_list: A list of 2D numpy arrays (matrices) to be encoded.
        max_cdr3: The maximum size for the first dimension to pad to.
        max_epi: The maximum size for the second dimension to pad to.

    Returns:
        A single numpy array containing the padded and stacked matrices.
    """
    padded = []
    for mat in mat_list:
        h, w = mat.shape
        pad_h = max_cdr3 - h
        pad_w = max_epi - w
        padded.append(np.pad(mat, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0))
    return np.stack(padded, axis=0)

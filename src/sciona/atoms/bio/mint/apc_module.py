from __future__ import annotations

import numpy as np

import icontract
from sciona.ghost.registry import register_atom
from .apc_module_witnesses import witness_apccoreevaluation

@register_atom(witness_apccoreevaluation)  # type: ignore[untyped-decorator]
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.require(lambda x: x.ndim >= 2, "x must have at least two dimensions")
@icontract.ensure(lambda result: result is not None, "ApcCoreEvaluation output must not be None")
def apccoreevaluation(x: np.ndarray) -> np.ndarray:
    """Apply average-product correction (APC) over the final two axes.

    Args:
        x: Array whose last two axes form the pairwise matrix to correct.

    Returns:
        Corrected array with the same shape as ``x``.
    """
    a1 = x.sum(axis=-1, keepdims=True)
    a2 = x.sum(axis=-2, keepdims=True)
    a12 = x.sum(axis=(-1, -2), keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.divide(
            a1 * a2,
            a12,
            out=np.zeros_like(x, dtype=np.result_type(x, np.float64)),
            where=a12 != 0,
        )
    return x - avg

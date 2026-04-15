from __future__ import annotations

from typing import Any
import numpy as np

import icontract
from sciona.ghost.registry import register_atom
from .apc_module_witnesses import witness_apccoreevaluation

@register_atom(witness_apccoreevaluation)  # type: ignore[untyped-decorator]
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.ensure(lambda result: result is not None, "ApcCoreEvaluation output must not be None")
def apccoreevaluation(x: np.ndarray) -> np.ndarray:
    """Executes the standalone Autoregressive Predictive Coding (APC) computation as a pure stateless function of the input.

    Args:
        x: Direct method argument; no stated structural constraints.

    Returns:
        Return value produced solely from x with no persistent state.
    """
    # Autoregressive Predictive Coding: predict next frame from context
    # Simple APC: use last-frame predictor (identity shift)
    if x.ndim == 1:
        return np.roll(x, -1)
    # For 2D+ input: shift along time axis (axis 0)
    return np.roll(x, -1, axis=0)

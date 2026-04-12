from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_addquantumlink(G: AbstractArray, node_A: AbstractArray, node_B: AbstractArray, chain_size: AbstractScalar) -> AbstractArray:
    """Shape-and-type check for add quantum link. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=G.shape,
        dtype="float64",)
    
    return result

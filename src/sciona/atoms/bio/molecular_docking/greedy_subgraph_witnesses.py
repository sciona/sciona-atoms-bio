from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal


def witness_greedy_maximum_subgraph(
    adjacency: AbstractArray,
    scores: AbstractArray,
) -> AbstractArray:
    """Describe the selected subgraph mask for greedy maximum-subgraph search."""
    _ = scores
    result = AbstractArray(
        shape=adjacency.shape,
        dtype="float64",)
    
    return result

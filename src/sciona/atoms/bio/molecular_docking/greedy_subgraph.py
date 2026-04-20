from __future__ import annotations

import icontract
import numpy as np
from sciona.ghost.registry import register_atom

from .greedy_subgraph_witnesses import witness_greedy_maximum_subgraph


@register_atom(witness_greedy_maximum_subgraph)
@icontract.require(lambda adjacency: adjacency is not None, "adjacency cannot be None")
@icontract.require(lambda adjacency: isinstance(adjacency, np.ndarray), "adjacency must be np.ndarray")
@icontract.require(lambda adjacency: adjacency.ndim == 2, "adjacency must be a 2D matrix")
@icontract.require(lambda adjacency: adjacency.shape[0] == adjacency.shape[1], "adjacency must be square")
@icontract.require(lambda scores: scores is not None, "scores cannot be None")
@icontract.require(lambda scores: isinstance(scores, np.ndarray), "scores must be np.ndarray")
@icontract.require(lambda scores: scores.ndim == 1, "scores must be a 1D vector")
@icontract.require(lambda adjacency, scores: adjacency.shape[0] == scores.shape[0], "scores length must match adjacency")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result.dtype == np.bool_, "result must be a boolean mask")
def greedy_maximum_subgraph(adjacency: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Return a greedy maximum-weight independent-set subgraph mask.

    Args:
        adjacency: Conflict adjacency matrix of the molecular interaction graph.
        scores: Non-negative node affinity scores used as independent-set weights.

    Returns:
        Boolean mask of selected nodes. No selected pair has an adjacency conflict.
    """
    n = int(adjacency.shape[0])
    conflicts = np.asarray(adjacency, dtype=bool).copy()
    np.fill_diagonal(conflicts, False)
    weights = np.asarray(scores, dtype=float)
    degrees = conflicts.sum(axis=1).astype(float)

    # The upstream classical helper orders by weight divided by one plus degree.
    priority = weights / (1.0 + degrees)
    order = sorted(range(n), key=lambda idx: (-priority[idx], -weights[idx], idx))

    selected = np.zeros(n, dtype=bool)
    for idx in order:
        if not np.any(conflicts[idx] & selected):
            selected[idx] = True
    return selected

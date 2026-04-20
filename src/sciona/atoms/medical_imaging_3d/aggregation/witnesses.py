from __future__ import annotations

"""Ghost witnesses for 3D medical-imaging aggregation atoms."""

from sciona.ghost.abstract import AbstractArray, AbstractScalar


def witness_case_probability_from_nodule_scores(
    nodule_probabilities: AbstractArray,
    baseline_logit: AbstractScalar | None = None,
) -> AbstractArray:
    """Model leaky noisy-or aggregation as one scalar probability per case."""
    if len(nodule_probabilities.shape) >= 2:
        cases = nodule_probabilities.shape[0]
    else:
        cases = 1
    return AbstractArray(shape=(int(cases),), dtype="float64")

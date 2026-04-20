from __future__ import annotations

"""Case-level aggregation atoms for 3D pulmonary nodule classifiers."""

import numpy as np
import icontract

from sciona.atoms.medical_imaging_3d.aggregation.witnesses import (
    witness_case_probability_from_nodule_scores,
)
from sciona.ghost.registry import register_atom


def _sigmoid(value: float) -> float:
    """Evaluate a numerically stable scalar logistic function."""
    if value >= 0.0:
        scale = np.exp(-value)
        return float(1.0 / (1.0 + scale))
    scale = np.exp(value)
    return float(scale / (1.0 + scale))


@register_atom(witness_case_probability_from_nodule_scores)
@icontract.require(
    lambda nodule_probabilities: isinstance(nodule_probabilities, np.ndarray),
    "nodule_probabilities must be a NumPy array",
)
@icontract.require(
    lambda nodule_probabilities: nodule_probabilities.ndim in {1, 2},
    "nodule_probabilities must be a one- or two-dimensional array",
)
@icontract.require(
    lambda nodule_probabilities: nodule_probabilities.size > 0,
    "nodule_probabilities must contain at least one score",
)
@icontract.require(
    lambda nodule_probabilities: bool(np.all(np.isfinite(nodule_probabilities))),
    "nodule_probabilities must be finite",
)
@icontract.require(
    lambda nodule_probabilities: bool(
        np.all((0.0 <= nodule_probabilities) & (nodule_probabilities <= 1.0))
    ),
    "nodule_probabilities must be probabilities in [0, 1]",
)
@icontract.require(lambda baseline_logit: np.isfinite(baseline_logit), "baseline_logit must be finite")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be a NumPy array")
@icontract.ensure(lambda result: result.ndim == 1, "result must contain one probability per case")
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))), "result probabilities must be finite")
@icontract.ensure(lambda result: bool(np.all((0.0 <= result) & (result <= 1.0))), "result must be in [0, 1]")
def case_probability_from_nodule_scores(
    nodule_probabilities: np.ndarray,
    baseline_logit: float = -30.0,
) -> np.ndarray:
    """Combine candidate nodule cancer probabilities into case probabilities.

    This implements the leaky noisy-or aggregation used by the DSB2017
    CaseNet classifier after per-nodule probabilities have already been
    computed. A one-dimensional input is treated as one case; a two-dimensional
    input is interpreted as ``cases x candidate_nodules``.
    """
    scores = np.asarray(nodule_probabilities, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    dummy_probability = _sigmoid(float(baseline_logit))
    missed_all_nodules = np.prod(1.0 - scores, axis=1)
    case_probabilities = 1.0 - missed_all_nodules * (1.0 - dummy_probability)
    return np.clip(case_probabilities, 0.0, 1.0)

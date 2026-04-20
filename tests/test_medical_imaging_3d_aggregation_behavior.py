from __future__ import annotations

import numpy as np
import pytest

from sciona.atoms.medical_imaging_3d.aggregation import (
    case_probability_from_nodule_scores,
)


def test_case_probability_matches_source_leaky_noisy_or_equation() -> None:
    scores = np.array(
        [
            [0.10, 0.20, 0.30],
            [0.00, 0.50, 0.75],
        ],
        dtype=np.float64,
    )
    baseline_logit = -2.0
    dummy_probability = 1.0 / (1.0 + np.exp(-baseline_logit))

    expected = 1.0 - np.prod(1.0 - scores, axis=1) * (1.0 - dummy_probability)

    result = case_probability_from_nodule_scores(scores, baseline_logit=baseline_logit)

    np.testing.assert_allclose(result, expected)
    assert result.shape == (2,)
    assert np.all((0.0 <= result) & (result <= 1.0))


def test_case_probability_treats_one_dimensional_scores_as_single_case() -> None:
    scores = np.array([0.10, 0.20, 0.30], dtype=np.float64)

    result = case_probability_from_nodule_scores(scores)

    expected = 1.0 - np.prod(1.0 - scores) * (1.0 - 1.0 / (1.0 + np.exp(30.0)))
    np.testing.assert_allclose(result, np.array([expected]))


def test_case_probability_rejects_non_probability_scores() -> None:
    with pytest.raises(Exception):
        case_probability_from_nodule_scores(np.array([[0.2, 1.2]], dtype=np.float64))

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from sciona.atoms.bio.molecular_docking.minimize_bandwidth.atoms import (
    aggregate_maximum_distance_as_bandwidth,
    build_sparse_graph_view,
    build_threshold_search_space,
    compute_absolute_weighted_index_distances,
    compute_symmetric_bandwidth_reducing_order,
    enforce_threshold_sparsity,
    enumerate_threshold_based_permutations,
    extract_final_permutation,
    initialize_reduction_state,
    propose_greedy_permutation_step,
    select_minimum_bandwidth_permutation,
    update_state_with_improvement_criterion,
    validate_square_matrix_shape,
)
from sciona.ghost.registry import REGISTRY


def _bandwidth(matrix: np.ndarray) -> int:
    rows, cols = np.nonzero(matrix)
    return int(np.max(np.abs(rows - cols))) if rows.size else 0


def _path_matrix_in_poor_order() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )


def test_shape_and_symmetric_validators_accept_and_reject_expected_inputs() -> None:
    square = np.eye(3)

    assert validate_square_matrix_shape(square) is square
    with pytest.raises(ValueError, match="Matrix must be square"):
        validate_square_matrix_shape(np.ones((2, 3)))

    dense_validator = REGISTRY["validate_symmetric_input_dense"]["impl"]
    thresholded_validator = REGISTRY["validate_symmetric_input_thresholded"]["impl"]
    asymmetric = np.array([[1.0, 2.0], [0.0, 1.0]])

    assert dense_validator(square) is square
    assert thresholded_validator(square) is square
    with pytest.raises(ValueError, match="not symmetric"):
        dense_validator(asymmetric)
    with pytest.raises(ValueError, match="not symmetric"):
        thresholded_validator(asymmetric)


def test_weighted_bandwidth_metric_uses_absolute_value_times_index_distance() -> None:
    matrix = np.array(
        [
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 4.0],
            [3.0, 0.0, 0.0],
        ]
    )

    distances = compute_absolute_weighted_index_distances(matrix)

    expected = np.array(
        [
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 4.0],
            [6.0, 0.0, 0.0],
        ]
    )
    assert np.array_equal(distances, expected)
    assert aggregate_maximum_distance_as_bandwidth(distances) == pytest.approx(6.0)
    assert aggregate_maximum_distance_as_bandwidth(np.array([])) == pytest.approx(0.0)


def test_initialize_reduction_state_copies_absolute_matrix_and_tracks_bandwidth() -> None:
    matrix = np.array([[0.0, -2.0, 0.0], [-2.0, 0.0, 5.0], [0.0, 5.0, 0.0]])

    state_array = initialize_reduction_state(matrix)
    state = state_array[0]

    assert state["accumulated_permutation"] == [0, 1, 2]
    assert state["remaining_iterations"] == 100
    assert state["bandwidth"] == 1
    assert np.array_equal(state["working_matrix"], np.abs(matrix))
    assert not np.shares_memory(state["working_matrix"], matrix)


def test_greedy_step_returns_rcm_candidate_and_measured_bandwidth() -> None:
    matrix = _path_matrix_in_poor_order()
    state = initialize_reduction_state(matrix)

    returned_state, permutation, candidate_matrix, candidate_bandwidth = propose_greedy_permutation_step(state)

    assert returned_state is state
    assert sorted(permutation) == [0, 1, 2, 3]
    assert np.array_equal(candidate_matrix, matrix[np.ix_(permutation, permutation)])
    assert candidate_bandwidth == _bandwidth(candidate_matrix)
    assert candidate_bandwidth <= state[0]["bandwidth"]


def test_update_state_accepts_only_improving_candidates_and_extracts_final_permutation() -> None:
    state = np.empty(1, dtype=object)
    state[0] = {
        "working_matrix": np.eye(3),
        "accumulated_permutation": [2, 0, 1],
        "bandwidth": 2,
        "remaining_iterations": 3,
    }
    candidate_matrix = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    improved_state, should_continue = update_state_with_improvement_criterion(
        state,
        candidate_permutation=[1, 2, 0],
        candidate_matrix=candidate_matrix,
        candidate_bandwidth=1,
    )

    assert should_continue is True
    assert improved_state[0]["bandwidth"] == 1
    assert improved_state[0]["remaining_iterations"] == 2
    assert improved_state[0]["accumulated_permutation"] == [0, 1, 2]
    assert extract_final_permutation(improved_state) == [0, 1, 2]

    unchanged_state, should_continue = update_state_with_improvement_criterion(
        improved_state,
        candidate_permutation=[2, 1, 0],
        candidate_matrix=np.eye(3),
        candidate_bandwidth=3,
    )

    assert should_continue is False
    assert unchanged_state[0]["bandwidth"] == 1
    assert unchanged_state[0]["accumulated_permutation"] == [0, 1, 2]


def test_threshold_sparsity_and_sparse_graph_view_preserve_shape_without_mutating_input() -> None:
    matrix = np.array([[0.0, 0.2, 0.8], [0.2, 0.0, -0.1], [0.8, -0.1, 0.0]])

    thresholded = enforce_threshold_sparsity(matrix, 0.25)
    sparse_view = build_sparse_graph_view(thresholded)

    assert np.array_equal(
        thresholded,
        np.array([[0.0, 0.0, 0.8], [0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]),
    )
    assert np.array_equal(
        matrix,
        np.array([[0.0, 0.2, 0.8], [0.2, 0.0, -0.1], [0.8, -0.1, 0.0]]),
    )
    assert np.array_equal(sparse_view, thresholded)
    assert not np.shares_memory(sparse_view, thresholded)


def test_rcm_order_matches_scipy_reverse_cuthill_mckee_for_symmetric_input() -> None:
    matrix = _path_matrix_in_poor_order()

    order = compute_symmetric_bandwidth_reducing_order(matrix)
    expected = reverse_cuthill_mckee(csr_matrix(matrix), symmetric_mode=True)

    assert np.array_equal(order, expected)
    assert sorted(order.tolist()) == [0, 1, 2, 3]


def test_threshold_search_space_and_enumeration_generate_valid_permutations() -> None:
    matrix = _path_matrix_in_poor_order()

    amplitude, truncation_values = build_threshold_search_space(matrix)
    permutations = enumerate_threshold_based_permutations(
        matrix,
        amplitude,
        np.array([0.1, 0.5], dtype=float),
    )

    assert amplitude == pytest.approx(1.0)
    assert truncation_values[0] == pytest.approx(0.1)
    assert truncation_values[-1] == pytest.approx(0.99)
    assert len(truncation_values) == 90
    assert permutations.shape == (2, 4)
    assert all(sorted(row.tolist()) == [0, 1, 2, 3] for row in permutations)


def test_select_minimum_bandwidth_permutation_prefers_lowest_bandwidth_candidate() -> None:
    matrix = _path_matrix_in_poor_order()
    candidates = np.array(
        [
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        ]
    )

    assert _bandwidth(matrix[np.ix_(candidates[0], candidates[0])]) == 3
    assert _bandwidth(matrix[np.ix_(candidates[1], candidates[1])]) == 1
    assert select_minimum_bandwidth_permutation(matrix, candidates) == [0, 3, 1, 2]
    assert select_minimum_bandwidth_permutation(matrix, np.empty((0, 4), dtype=int)) == [0, 1, 2, 3]

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal


def witness_validate_square_matrix_shape(mat: AbstractArray) -> AbstractArray:
    """Describe the validated square matrix passed into bandwidth minimization."""
    result = AbstractArray(
        shape=mat.shape,
        dtype="float64",)

    return result

def witness_compute_absolute_weighted_index_distances(square_mat: AbstractArray) -> AbstractArray:
    """Shape-and-type check for compute absolute weighted index distances. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=square_mat.shape,
        dtype="float64",)

    return result

def witness_aggregate_maximum_distance_as_bandwidth(weighted_distances: AbstractArray) -> AbstractArray:
    """Shape-and-type check for aggregate maximum distance as bandwidth. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=weighted_distances.shape,
        dtype="float64",)

    return result

def witness_validate_symmetric_input(matrix: AbstractArray) -> AbstractArray:
    """Shape-and-type check for validate symmetric input. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=matrix.shape,
        dtype="float64",)

    return result

def witness_initialize_reduction_state(symmetric_matrix: AbstractArray) -> AbstractArray:
    """Shape-and-type check for initialize reduction state. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=symmetric_matrix.shape,
        dtype="float64",)

    return result

def witness_propose_greedy_permutation_step(iteration_state: AbstractArray) -> AbstractArray:
    """Shape-and-type check for propose greedy permutation step. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=iteration_state.shape,
        dtype="float64",)

    return result

def witness_update_state_with_improvement_criterion(current_iteration_state: AbstractArray, candidate_permutation: AbstractArray, candidate_matrix: AbstractArray, candidate_bandwidth: AbstractArray) -> AbstractArray:
    """Shape-and-type check for update state with improvement criterion. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=current_iteration_state.shape,
        dtype="float64",)

    return result

def witness_extract_final_permutation(terminal_state: AbstractArray) -> AbstractArray:
    """Shape-and-type check for extract final permutation. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=terminal_state.shape,
        dtype="float64",)

    return result

def witness_enforce_threshold_sparsity(mat: AbstractArray, threshold: AbstractArray) -> AbstractArray:
    """Shape-and-type check for enforce threshold sparsity. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=mat.shape,
        dtype="float64",)

    return result

def witness_build_sparse_graph_view(thresholded_matrix: AbstractArray) -> AbstractArray:
    """Shape-and-type check for build sparse graph view. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=thresholded_matrix.shape,
        dtype="float64",)

    return result

def witness_compute_symmetric_bandwidth_reducing_order(sparse_matrix: AbstractArray) -> AbstractArray:
    """Shape-and-type check for compute symmetric bandwidth reducing order. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=sparse_matrix.shape,
        dtype="float64",)

    return result

def witness_build_threshold_search_space(validated_mat: AbstractArray) -> AbstractArray:
    """Shape-and-type check for build threshold search space. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=validated_mat.shape,
        dtype="float64",)

    return result

def witness_enumerate_threshold_based_permutations(validated_mat: AbstractArray, mat_amplitude: AbstractArray, truncation_values: AbstractArray) -> AbstractArray:
    """Shape-and-type check for enumerate threshold based permutations. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=validated_mat.shape,
        dtype="float64",)

    return result

def witness_select_minimum_bandwidth_permutation(validated_mat: AbstractArray, candidate_permutations: AbstractArray) -> AbstractArray:
    """Shape-and-type check for select minimum bandwidth permutation. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=validated_mat.shape,
        dtype="float64",)

    return result

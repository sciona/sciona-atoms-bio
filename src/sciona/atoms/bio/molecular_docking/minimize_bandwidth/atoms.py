from __future__ import annotations
"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_aggregate_maximum_distance_as_bandwidth,
    witness_build_sparse_graph_view,
    witness_build_threshold_search_space,
    witness_compute_absolute_weighted_index_distances,
    witness_compute_symmetric_bandwidth_reducing_order,
    witness_enforce_threshold_sparsity,
    witness_enumerate_threshold_based_permutations,
    witness_extract_final_permutation,
    witness_initialize_reduction_state,
    witness_propose_greedy_permutation_step,
    witness_select_minimum_bandwidth_permutation,
    witness_update_state_with_improvement_criterion,
    witness_validate_square_matrix_shape,
    witness_validate_symmetric_input
)
def _matrix_bandwidth(mat: np.ndarray) -> int:
    """Compute the bandwidth of a matrix: max |i-j| where mat[i,j] != 0."""
    rows, cols = np.nonzero(mat)
    if len(rows) == 0:
        return 0
    return int(np.max(np.abs(rows - cols)))


def _permute_matrix(mat: np.ndarray, perm: list[int]) -> np.ndarray:
    """Permute rows and columns of a matrix according to perm."""
    p = np.array(perm)
    return mat[np.ix_(p, p)]


def _minimize_bandwidth_rcm(mat: np.ndarray) -> np.ndarray:
    """Compute reverse Cuthill-McKee ordering for bandwidth reduction."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    sparse = csr_matrix(mat)
    return reverse_cuthill_mckee(sparse, symmetric_mode=True)


@register_atom(witness_validate_square_matrix_shape)
@icontract.require(lambda mat: mat is not None, "mat cannot be None")
@icontract.require(lambda mat: isinstance(mat, np.ndarray), "mat must be np.ndarray")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def validate_square_matrix_shape(mat: np.ndarray) -> np.ndarray:
    """Check that the input matrix has identical row and column counts; fail if not square.

    Args:
        mat: must have shape (n, n)

    Returns:
        square_mat: same data as input, guaranteed square
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {mat.shape}")
    return mat


@register_atom(witness_compute_absolute_weighted_index_distances)
@icontract.require(lambda square_mat: square_mat is not None, "square_mat cannot be None")
@icontract.require(lambda square_mat: isinstance(square_mat, np.ndarray), "square_mat must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def compute_absolute_weighted_index_distances(square_mat: np.ndarray) -> np.ndarray:
    """Enumerate all entries and compute absolute weighted index distance for each element as abs(value * (row_index - col_index)).

    Args:
        square_mat: square matrix

    Returns:
        weighted_distances: non-negative, one value per matrix entry
    """
    n = square_mat.shape[0]
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    return np.abs(square_mat * (rows - cols))


@register_atom(witness_aggregate_maximum_distance_as_bandwidth)
@icontract.require(lambda weighted_distances: weighted_distances is not None, "weighted_distances cannot be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def aggregate_maximum_distance_as_bandwidth(weighted_distances: np.ndarray) -> float:
    """Take the maximum weighted distance and convert it to float as the final bandwidth metric.

    Args:
        weighted_distances: non-empty if matrix has at least one element

    Returns:
        bandwidth: >= 0
    """
    arr = np.asarray(weighted_distances)
    return float(arr.max()) if arr.size > 0 else 0.0


@register_atom(witness_validate_symmetric_input, name="validate_symmetric_input_dense")
@icontract.require(lambda matrix: matrix is not None, "matrix cannot be None")
@icontract.require(lambda matrix: isinstance(matrix, np.ndarray), "matrix must be np.ndarray")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def validate_symmetric_input(matrix: np.ndarray) -> np.ndarray:
    """Verify the matrix is numerically symmetric within tolerance before any reordering logic proceeds.

    Args:
        matrix: Must satisfy allclose(matrix, matrix.T, atol=1e-8)

    Returns:
        symmetric_matrix: Same values as input; guaranteed symmetric or function fails
    """
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValueError("Matrix is not symmetric within tolerance 1e-8")
    return matrix


@register_atom(witness_initialize_reduction_state)
@icontract.require(lambda symmetric_matrix: symmetric_matrix is not None, "symmetric_matrix cannot be None")
@icontract.require(lambda symmetric_matrix: isinstance(symmetric_matrix, np.ndarray), "symmetric_matrix must be np.ndarray")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def initialize_reduction_state(symmetric_matrix: np.ndarray) -> np.ndarray:
    """Create the sanitized working matrix and initial optimization state for greedy bandwidth descent.

    Args:
        symmetric_matrix: Symmetric numeric matrix

    Returns:
        iteration_state: working_matrix=abs(copy), accumulated_permutation=identity permutation, bandwidth=matrix_bandwidth(working_matrix), remaining_iterations=100
    """
    working_matrix = np.abs(symmetric_matrix.copy())
    n = working_matrix.shape[0]
    acc_perm = list(range(n))
    # Compute bandwidth: max |i - j| for non-zero entries
    rows, cols = np.nonzero(working_matrix)
    bw = int(np.max(np.abs(rows - cols))) if len(rows) > 0 else 0
    state = np.empty(1, dtype=object)
    state[0] = {
        'working_matrix': working_matrix,
        'accumulated_permutation': acc_perm,
        'bandwidth': bw,
        'remaining_iterations': 100,
    }
    return state


@register_atom(witness_propose_greedy_permutation_step)
@icontract.require(lambda iteration_state: iteration_state is not None, "iteration_state cannot be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def propose_greedy_permutation_step(iteration_state: np.ndarray | dict[str, np.ndarray | list[int] | int]) -> tuple[np.ndarray | dict[str, np.ndarray | list[int] | int], list[int], np.ndarray, int]:
    """From the current working matrix, compute one candidate permutation and its resulting matrix bandwidth.

    Args:
        iteration_state: Contains current working_matrix and associated state

    Returns:
        current_iteration_state: Pass-through of input state for synchronized evaluation
        candidate_permutation: Produced by minimize_bandwidth_global on current working_matrix
        candidate_matrix: permute_matrix(current working_matrix, candidate_permutation)
        candidate_bandwidth: matrix_bandwidth(candidate_matrix)
    """
    state = iteration_state[0] if hasattr(iteration_state, '__getitem__') else iteration_state
    wm = state['working_matrix']
    candidate_perm = list(_minimize_bandwidth_rcm(wm))
    candidate_matrix = _permute_matrix(wm, candidate_perm)
    candidate_bw = _matrix_bandwidth(candidate_matrix)
    return iteration_state, candidate_perm, candidate_matrix, candidate_bw


@register_atom(witness_update_state_with_improvement_criterion)
@icontract.require(lambda current_iteration_state: current_iteration_state is not None, "current_iteration_state cannot be None")
@icontract.require(lambda candidate_permutation: candidate_permutation is not None, "candidate_permutation cannot be None")
@icontract.require(lambda candidate_matrix: candidate_matrix is not None, "candidate_matrix cannot be None")
@icontract.require(lambda candidate_matrix: isinstance(candidate_matrix, np.ndarray), "candidate_matrix must be np.ndarray")
@icontract.require(lambda candidate_bandwidth: candidate_bandwidth is not None, "candidate_bandwidth cannot be None")
@icontract.require(lambda candidate_bandwidth: isinstance(candidate_bandwidth, (int,)), "candidate_bandwidth must be int")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def update_state_with_improvement_criterion(current_iteration_state: np.ndarray | dict[str, np.ndarray | list[int] | int], candidate_permutation: list[int], candidate_matrix: np.ndarray, candidate_bandwidth: int) -> tuple[np.ndarray, bool]:
    """Apply iteration budget checks, compare candidate vs current bandwidth, and either accept the candidate state or terminate without update.

    Args:
        current_iteration_state: Contains working_matrix, accumulated_permutation, bandwidth, remaining_iterations
        candidate_permutation: Permutation proposed for this iteration
        candidate_matrix: Permuted working matrix for this iteration
        candidate_bandwidth: Bandwidth of candidate_matrix

    Returns:
        next_iteration_state: Counter decremented; if candidate improves bandwidth, state is updated, otherwise state remains effectively terminal
        continue_search: True only when candidate_bandwidth < current bandwidth and budget not exhausted
    """
    state = current_iteration_state[0] if hasattr(current_iteration_state, '__getitem__') else current_iteration_state
    remaining = state['remaining_iterations'] - 1
    current_bw = state['bandwidth']

    if candidate_bandwidth < current_bw and remaining > 0:
        # Compose permutations
        old_perm = state['accumulated_permutation']
        new_acc = [old_perm[i] for i in candidate_permutation]
        new_state = np.empty(1, dtype=object)
        new_state[0] = {
            'working_matrix': candidate_matrix,
            'accumulated_permutation': new_acc,
            'bandwidth': candidate_bandwidth,
            'remaining_iterations': remaining,
        }
        return (new_state, True)
    else:
        new_state = np.empty(1, dtype=object)
        new_state[0] = {
            'working_matrix': state['working_matrix'],
            'accumulated_permutation': state['accumulated_permutation'],
            'bandwidth': current_bw,
            'remaining_iterations': remaining,
        }
        return (new_state, False)


@register_atom(witness_extract_final_permutation)
@icontract.require(lambda terminal_state: terminal_state is not None, "terminal_state cannot be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def extract_final_permutation(terminal_state: np.ndarray | dict[str, np.ndarray | list[int] | int]) -> list[int]:
    """Return the accumulated permutation from the terminal state as the function result.

    Args:
        terminal_state: State after greedy loop termination

    Returns:
        acc_permutation: Final accumulated row/column reordering permutation
    """
    state = terminal_state[0] if hasattr(terminal_state, '__getitem__') else terminal_state
    return list(state['accumulated_permutation'])


@register_atom(witness_enforce_threshold_sparsity)
@icontract.require(lambda mat: mat is not None, "mat cannot be None")
@icontract.require(lambda mat: isinstance(mat, np.ndarray), "mat must be np.ndarray")
@icontract.require(lambda threshold: threshold is not None, "threshold cannot be None")
@icontract.require(lambda threshold: isinstance(threshold, (float,)), "threshold must be float")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def enforce_threshold_sparsity(mat: np.ndarray, threshold: float) -> np.ndarray:
    """Create a thresholded copy of the input matrix where all entries below the threshold are set to zero, preserving the original input matrix.

    Args:
        mat: Square, symmetric expected by downstream ordering step
        threshold: Values strictly below this are zeroed

    Returns:
        thresholded_matrix: Same shape as mat; entries < threshold replaced with 0
    """
    result = mat.copy()
    result[np.abs(result) < threshold] = 0.0
    return result


@register_atom(witness_build_sparse_graph_view)
@icontract.require(lambda thresholded_matrix: thresholded_matrix is not None, "thresholded_matrix cannot be None")
@icontract.require(lambda thresholded_matrix: isinstance(thresholded_matrix, np.ndarray), "thresholded_matrix must be np.ndarray")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def build_sparse_graph_view(thresholded_matrix: np.ndarray) -> np.ndarray:
    """Convert the thresholded dense matrix into a Compressed Sparse Row (CSR) representation suitable for graph-based bandwidth reduction.

    Args:
        thresholded_matrix: Numeric matrix; typically sparse after thresholding

    Returns:
        sparse_matrix: Equivalent numeric structure to thresholded_matrix
    """
    # Return as-is since downstream expects np.ndarray; sparsity is implicit
    return thresholded_matrix.copy()


@register_atom(witness_compute_symmetric_bandwidth_reducing_order)
@icontract.require(lambda sparse_matrix: sparse_matrix is not None, "sparse_matrix cannot be None")
@icontract.require(lambda sparse_matrix: isinstance(sparse_matrix, np.ndarray), "sparse_matrix must be np.ndarray")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def compute_symmetric_bandwidth_reducing_order(sparse_matrix: np.ndarray) -> np.ndarray:
    """Run reverse Cuthill-McKee in symmetric mode on the sparse matrix and materialize the permutation as a NumPy array.

    Args:
        sparse_matrix: Represents a symmetric graph structure

    Returns:
        reordered_matrix: Permutation of row/column indices for reduced bandwidth
    """
    return np.asarray(_minimize_bandwidth_rcm(sparse_matrix))


@register_atom(witness_validate_symmetric_input, name="validate_symmetric_input_thresholded")
@icontract.require(lambda mat: mat is not None, "mat cannot be None")
@icontract.require(lambda mat: isinstance(mat, np.ndarray), "mat must be np.ndarray")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def validate_symmetric_input(mat: np.ndarray) -> np.ndarray:
    """Verify the input matrix is symmetric within tolerance and fail fast if the precondition is violated.

    Args:
        mat: square; expected symmetric

    Returns:
        validated_mat: symmetric within atol=1e-8
    """
    if not np.allclose(mat, mat.T, atol=1e-8):
        raise ValueError("Matrix is not symmetric within tolerance 1e-8")
    return mat


@register_atom(witness_build_threshold_search_space)
@icontract.require(lambda validated_mat: validated_mat is not None, "validated_mat cannot be None")
@icontract.require(lambda validated_mat: isinstance(validated_mat, np.ndarray), "validated_mat must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def build_threshold_search_space(validated_mat: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute matrix amplitude from absolute values and construct the truncation sweep used to explore candidate permutations.

    Args:
        validated_mat: symmetric

    Returns:
        mat_amplitude: ptp(abs(validated_mat).ravel())
        truncation_values: range [0.1, 1.0) with step 0.01
    """
    mat_amplitude = float(np.ptp(np.abs(validated_mat).ravel()))
    truncation_values = np.arange(0.1, 1.0, 0.01)
    return mat_amplitude, truncation_values


@register_atom(witness_enumerate_threshold_based_permutations)
@icontract.require(lambda validated_mat: validated_mat is not None, "validated_mat cannot be None")
@icontract.require(lambda validated_mat: isinstance(validated_mat, np.ndarray), "validated_mat must be np.ndarray")
@icontract.require(lambda mat_amplitude: mat_amplitude is not None, "mat_amplitude cannot be None")
@icontract.require(lambda mat_amplitude: isinstance(mat_amplitude, (float,)), "mat_amplitude must be float")
@icontract.require(lambda truncation_values: truncation_values is not None, "truncation_values cannot be None")
@icontract.require(lambda truncation_values: isinstance(truncation_values, np.ndarray), "truncation_values must be np.ndarray")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def enumerate_threshold_based_permutations(validated_mat: np.ndarray, mat_amplitude: float, truncation_values: np.ndarray) -> np.ndarray:
    """Generate candidate permutations by calling thresholded bandwidth minimization for each scaled truncation value.

    Args:
        validated_mat: symmetric
        mat_amplitude: non-negative
        truncation_values: finite values in [0.1, 1.0)

    Returns:
        candidate_permutations: each item is a valid row/column permutation
    """
    perms = []
    for tv in truncation_values:
        threshold = tv * mat_amplitude
        thresholded = validated_mat.copy()
        thresholded[np.abs(thresholded) < threshold] = 0.0
        perm = _minimize_bandwidth_rcm(thresholded)
        perms.append(perm)
    return np.array(perms)


@register_atom(witness_select_minimum_bandwidth_permutation)
@icontract.require(lambda validated_mat: validated_mat is not None, "validated_mat cannot be None")
@icontract.require(lambda validated_mat: isinstance(validated_mat, np.ndarray), "validated_mat must be np.ndarray")
@icontract.require(lambda candidate_permutations: candidate_permutations is not None, "candidate_permutations cannot be None")
@icontract.require(lambda candidate_permutations: isinstance(candidate_permutations, np.ndarray), "candidate_permutations must be np.ndarray")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def select_minimum_bandwidth_permutation(validated_mat: np.ndarray, candidate_permutations: np.ndarray) -> list[int]:
    """Evaluate each candidate by permuting the matrix and computing resulting bandwidth, then return the best permutation as a Python list.

    Args:
        validated_mat: symmetric
        candidate_permutations: non-empty iterable of valid permutations

    Returns:
        optimal_permutation: argmin over matrix_bandwidth(permute_matrix(validated_mat, perm))
    """
    best_perm = None
    best_bw = float('inf')
    for perm in candidate_permutations:
        perm_list = list(perm)
        permuted = _permute_matrix(validated_mat, perm_list)
        bw = _matrix_bandwidth(permuted)
        if bw < best_bw:
            best_bw = bw
            best_perm = perm_list
    return best_perm if best_perm is not None else list(range(validated_mat.shape[0]))

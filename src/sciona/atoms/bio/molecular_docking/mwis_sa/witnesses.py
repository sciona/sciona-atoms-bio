from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar


def witness_load_graphs_from_folder(folder_path: AbstractScalar) -> AbstractArray:
    """Shape-and-type check for load graphs from folder. Returns output metadata without running the real computation."""
    _ = folder_path
    result = AbstractArray(
        shape=(1,),
        dtype="float64",)

    return result

def witness_is_independent_set(graph: AbstractArray, subset: AbstractArray) -> AbstractScalar:
    """Shape-and-type check for is independent set. Returns output metadata without running the real computation."""
    _ = (graph, subset)
    result = AbstractScalar(dtype="bool")

    return result

def witness_calculate_weight(graph: AbstractArray, node_list: AbstractArray) -> AbstractScalar:
    """Shape-and-type check for calculate weight. Returns output metadata without running the real computation."""
    _ = (graph, node_list)
    result = AbstractScalar(dtype="float64")

    return result

def witness_to_qubo(graph: AbstractArray, penalty: AbstractScalar) -> AbstractArray:
    """Shape-and-type check for to qubo. Returns output metadata without running the real computation."""
    _ = penalty
    result = AbstractArray(
        shape=graph.shape,
        dtype="float64",)

    return result

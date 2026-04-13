from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_load_graphs_from_folder(folder_path: AbstractArray, *args, **kwargs) -> AbstractArray:
    """Shape-and-type check for load graphs from folder. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=(1,),
        dtype="float64",)

    return result

def witness_is_independent_set(graph: AbstractArray, subset: AbstractArray) -> AbstractArray:
    """Shape-and-type check for is independent set. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=graph.shape,
        dtype="float64",)

    return result

def witness_calculate_weight(graph: AbstractArray, node_list: AbstractArray) -> AbstractArray:
    """Shape-and-type check for calculate weight. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=graph.shape,
        dtype="float64",)

    return result

def witness_to_qubo(graph: AbstractArray, penalty: AbstractArray) -> AbstractArray:
    """Shape-and-type check for to qubo. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=graph.shape,
        dtype="float64",)

    return result

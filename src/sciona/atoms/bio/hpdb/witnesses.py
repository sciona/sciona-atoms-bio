from __future__ import annotations
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_iterate_pdb_atoms(element: AbstractArray) -> AbstractArray:
    """Shape-and-type check for iterate pdb atoms. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=element.shape,
        dtype="float64",
    )
    return result

def witness_iterate_pdb_residues(element: AbstractArray) -> AbstractArray:
    """Shape-and-type check for iterate pdb residues. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=element.shape,
        dtype="float64",
    )
    return result

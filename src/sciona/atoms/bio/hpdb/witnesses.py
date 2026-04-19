from __future__ import annotations

"""Ghost witnesses for hPDB hierarchy traversal atoms."""

from sciona.ghost.abstract import AbstractArray, AbstractScalar


def witness_iterate_pdb_atoms(
    pdb_element: AbstractScalar,
    chemical_element: AbstractScalar | None = None,
) -> AbstractArray:
    """Model atom traversal as producing a one-dimensional collection."""
    upper = pdb_element.max_val
    return AbstractArray(
        shape=(int(upper) if upper is not None and upper >= 0 else 0,),
        dtype="object",
    )


def witness_iterate_pdb_residues(
    pdb_element: AbstractScalar,
    chemical_element: AbstractScalar | None = None,
) -> AbstractArray:
    """Model residue traversal as producing a one-dimensional collection."""
    upper = pdb_element.max_val
    return AbstractArray(
        shape=(int(upper) if upper is not None and upper >= 0 else 0,),
        dtype="object",
    )

from __future__ import annotations

"""Source-aligned traversal atoms for hPDB-style protein structures."""

from dataclasses import dataclass

import icontract

from sciona.atoms.bio.hpdb.witnesses import (
    witness_iterate_pdb_atoms,
    witness_iterate_pdb_residues,
)
from sciona.ghost.registry import register_atom


@dataclass(frozen=True)
class Atom:
    """A single atom record in a PDB hierarchy."""

    name: str
    element: str


@dataclass(frozen=True)
class Residue:
    """A residue containing one or more atom records."""

    name: str
    atoms: tuple[Atom, ...]


@dataclass(frozen=True)
class Chain:
    """A chain containing ordered residues."""

    chain_id: str
    residues: tuple[Residue, ...]


@dataclass(frozen=True)
class Model:
    """A model containing one or more chains."""

    model_id: int
    chains: tuple[Chain, ...]


@dataclass(frozen=True)
class Structure:
    """A PDB structure containing one or more models."""

    models: tuple[Model, ...]


PDBElement = Atom | Residue | Chain | Model | Structure


def _element_matches(atom: Atom, chemical_element: str | None) -> bool:
    if chemical_element is None:
        return True
    return atom.element.upper() == chemical_element.upper()


@register_atom(witness_iterate_pdb_atoms)
@icontract.require(
    lambda pdb_element: isinstance(pdb_element, (Atom, Residue, Chain, Model, Structure)),
    "pdb_element must be an hPDB-style hierarchy object",
)
@icontract.require(
    lambda chemical_element: chemical_element is None or bool(chemical_element.strip()),
    "chemical_element must be None or a non-empty symbol",
)
@icontract.ensure(lambda result: all(isinstance(atom, Atom) for atom in result), "result must contain Atom records")
def iterate_pdb_atoms(
    pdb_element: PDBElement,
    chemical_element: str | None = None,
) -> tuple[Atom, ...]:
    """Return atoms reached by recursively walking an hPDB-style hierarchy.

    The traversal follows the local hPDB source model: atom, residue, chain,
    model, and structure objects are recursively expanded until atom records
    are reached. When `chemical_element` is provided, only atoms with that
    element symbol are returned.
    """
    if isinstance(pdb_element, Atom):
        return (pdb_element,) if _element_matches(pdb_element, chemical_element) else ()
    if isinstance(pdb_element, Residue):
        return tuple(atom for atom in pdb_element.atoms if _element_matches(atom, chemical_element))
    if isinstance(pdb_element, Chain):
        return tuple(
            atom
            for residue in pdb_element.residues
            for atom in iterate_pdb_atoms(residue, chemical_element)
        )
    if isinstance(pdb_element, Model):
        return tuple(
            atom
            for chain in pdb_element.chains
            for atom in iterate_pdb_atoms(chain, chemical_element)
        )
    return tuple(
        atom
        for model in pdb_element.models
        for atom in iterate_pdb_atoms(model, chemical_element)
    )


@register_atom(witness_iterate_pdb_residues)
@icontract.require(
    lambda pdb_element: isinstance(pdb_element, (Residue, Chain, Model, Structure)),
    "pdb_element must be a residue, chain, model, or structure",
)
@icontract.require(
    lambda chemical_element: chemical_element is None or bool(chemical_element.strip()),
    "chemical_element must be None or a non-empty symbol",
)
@icontract.ensure(
    lambda result: all(isinstance(residue, Residue) for residue in result),
    "result must contain Residue records",
)
def iterate_pdb_residues(
    pdb_element: Residue | Chain | Model | Structure,
    chemical_element: str | None = None,
) -> tuple[Residue, ...]:
    """Return residues reached by recursively walking an hPDB-style hierarchy.

    The traversal follows the local hPDB source model for residues under
    chain, model, and structure objects. When `chemical_element` is provided,
    only residues containing at least one matching atom are returned.
    """
    if isinstance(pdb_element, Residue):
        return (
            (pdb_element,)
            if any(_element_matches(atom, chemical_element) for atom in pdb_element.atoms)
            else ()
        )
    if isinstance(pdb_element, Chain):
        return tuple(
            residue
            for residue in pdb_element.residues
            if any(_element_matches(atom, chemical_element) for atom in residue.atoms)
        )
    if isinstance(pdb_element, Model):
        return tuple(
            residue
            for chain in pdb_element.chains
            for residue in iterate_pdb_residues(chain, chemical_element)
        )
    return tuple(
        residue
        for model in pdb_element.models
        for residue in iterate_pdb_residues(model, chemical_element)
    )

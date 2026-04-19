from __future__ import annotations

import pytest

from sciona.atoms.bio.hpdb import (
    Atom,
    Chain,
    Model,
    Residue,
    Structure,
    iterate_pdb_atoms,
    iterate_pdb_residues,
)


def _fixture_structure() -> Structure:
    gly = Residue(
        name="GLY",
        atoms=(
            Atom(name="N", element="N"),
            Atom(name="CA", element="C"),
            Atom(name="O", element="O"),
        ),
    )
    ser = Residue(
        name="SER",
        atoms=(
            Atom(name="N", element="N"),
            Atom(name="CB", element="C"),
        ),
    )
    chain = Chain(chain_id="A", residues=(gly, ser))
    return Structure(models=(Model(model_id=0, chains=(chain,)),))


def test_iterate_pdb_atoms_walks_full_hierarchy_in_order() -> None:
    atoms = iterate_pdb_atoms(_fixture_structure())

    assert [atom.name for atom in atoms] == ["N", "CA", "O", "N", "CB"]


def test_iterate_pdb_atoms_filters_by_element_case_insensitively() -> None:
    atoms = iterate_pdb_atoms(_fixture_structure(), "c")

    assert [atom.name for atom in atoms] == ["CA", "CB"]


def test_iterate_pdb_residues_walks_full_hierarchy_in_order() -> None:
    residues = iterate_pdb_residues(_fixture_structure())

    assert [residue.name for residue in residues] == ["GLY", "SER"]


def test_iterate_pdb_residues_filters_by_contained_atom_element() -> None:
    residues = iterate_pdb_residues(_fixture_structure(), "O")

    assert [residue.name for residue in residues] == ["GLY"]


def test_empty_element_filter_is_rejected() -> None:
    with pytest.raises(Exception, match="chemical_element must be None or a non-empty symbol"):
        iterate_pdb_atoms(_fixture_structure(), "")

    with pytest.raises(Exception, match="chemical_element must be None or a non-empty symbol"):
        iterate_pdb_residues(_fixture_structure(), " ")

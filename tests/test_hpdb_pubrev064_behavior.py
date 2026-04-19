from __future__ import annotations

import pytest

from sciona.atoms.bio.hpdb.atoms import iterate_pdb_atoms, iterate_pdb_residues


def test_pubrev064_hpdb_iterators_are_currently_empty_stubs() -> None:
    assert list(iterate_pdb_atoms("C")) == []
    assert list(iterate_pdb_atoms("N")) == []
    assert list(iterate_pdb_residues("C")) == []
    assert list(iterate_pdb_residues("N")) == []


def test_pubrev064_hpdb_optional_element_contract_drift_is_visible() -> None:
    with pytest.raises(Exception, match="element cannot be None"):
        list(iterate_pdb_atoms(None))

    with pytest.raises(Exception, match="element cannot be None"):
        list(iterate_pdb_residues(None))

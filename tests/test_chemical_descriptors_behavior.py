from __future__ import annotations

import numpy as np
import pytest

from sciona.atoms.bio.chemical_descriptors import (
    DEFAULT_DESCRIPTOR_NAMES,
    compute_descriptors,
    maccs_keys,
    morgan_fingerprint,
    smiles_to_mol,
)


pytest.importorskip("rdkit")


def test_smiles_to_mol_parses_ethanol_with_expected_graph_size() -> None:
    mol = smiles_to_mol("CCO")

    assert mol.GetNumHeavyAtoms() == 3
    assert mol.GetNumBonds() == 2


def test_smiles_to_mol_rejects_invalid_smiles() -> None:
    with pytest.raises(ValueError, match="invalid SMILES"):
        smiles_to_mol("C1CC")


def test_compute_descriptors_returns_finite_default_vector() -> None:
    descriptors = compute_descriptors(smiles_to_mol("CCO"))

    assert descriptors.shape == (len(DEFAULT_DESCRIPTOR_NAMES),)
    assert descriptors.dtype == np.float64
    assert np.all(np.isfinite(descriptors))


def test_compute_descriptors_accepts_explicit_descriptor_subset() -> None:
    descriptors = compute_descriptors(smiles_to_mol("CCO"), ["MolWt", "NumHDonors"])

    assert descriptors.shape == (2,)
    assert descriptors[0] > 40.0
    assert descriptors[1] == 1.0


def test_fingerprint_atoms_return_binary_vectors() -> None:
    mol = smiles_to_mol("CCO")
    morgan = morgan_fingerprint(mol, radius=2, n_bits=128)
    maccs = maccs_keys(mol)

    assert morgan.shape == (128,)
    assert maccs.shape == (166,)
    assert set(np.unique(morgan)).issubset({0, 1})
    assert set(np.unique(maccs)).issubset({0, 1})
    assert int(morgan.sum()) > 0
    assert int(maccs.sum()) > 0


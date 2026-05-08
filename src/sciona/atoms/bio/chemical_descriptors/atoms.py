from __future__ import annotations

"""RDKit-backed atoms for SMILES parsing and chemical feature extraction."""

from collections.abc import Sequence

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.atoms.bio.chemical_descriptors.witnesses import (
    witness_compute_descriptors,
    witness_maccs_keys,
    witness_morgan_fingerprint,
    witness_smiles_to_mol,
)
from sciona.ghost.registry import register_atom

DEFAULT_DESCRIPTOR_NAMES: tuple[str, ...] = (
    "MolLogP",
    "TPSA",
    "MolWt",
    "ExactMolWt",
    "HeavyAtomCount",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "BertzCT",
    "BalabanJ",
    "NumAromaticRings",
    "NumAliphaticRings",
    "FractionCSP3",
    "MaxEStateIndex",
    "MinEStateIndex",
    "MaxPartialCharge",
    "MinPartialCharge",
    "NumValenceElectrons",
    "NumRadicalElectrons",
    "FpDensityMorgan1",
)

def _load_rdkit_module(module_name: str) -> ModuleType:
    from importlib import import_module
    from types import ModuleType
    try:
        return import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for chemical descriptor atoms; install rdkit in the "
            "sciona matcher environment before calling this function."
        ) from exc

def _has_heavy_atoms(mol: object) -> bool:
    get_num_heavy_atoms = getattr(mol, "GetNumHeavyAtoms", None)
    return callable(get_num_heavy_atoms) and int(get_num_heavy_atoms()) > 0

def _has_valid_mol_surface(mol: object) -> bool:
    get_num_atoms = getattr(mol, "GetNumAtoms", None)
    return callable(get_num_atoms) and int(get_num_atoms()) > 0

def _descriptor_tuple(descriptor_names: Sequence[str] | None) -> tuple[str, ...]:
    if descriptor_names is None:
        return DEFAULT_DESCRIPTOR_NAMES
    return tuple(descriptor_names)

@register_atom(witness_smiles_to_mol)
@icontract.require(lambda smiles: bool(smiles.strip()), "smiles must be a non-empty string")
@icontract.ensure(lambda result: _has_heavy_atoms(result), "result must contain at least one heavy atom")
def smiles_to_mol(smiles: str) -> object:
    """Parse a SMILES string into a sanitized RDKit molecule graph.

    RDKit performs syntax checking, valence checks, implicit hydrogen handling,
    and aromaticity perception during parsing. Invalid strings are rejected with
    a ValueError so downstream atoms only receive molecule-like objects.
    """
    chem = _load_rdkit_module("rdkit.Chem")
    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES string: {smiles!r}")
    return mol

@register_atom(witness_compute_descriptors)
@icontract.require(lambda mol: _has_valid_mol_surface(mol), "mol must be an RDKit molecule")
@icontract.require(
    lambda descriptor_names: descriptor_names is None
    or (len(descriptor_names) > 0 and all(bool(name.strip()) for name in descriptor_names)),
    "descriptor_names must be None or a non-empty sequence of descriptor names",
)
@icontract.ensure(lambda result: result.ndim == 1, "result must be a one-dimensional descriptor vector")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "descriptor values must be finite")
@icontract.ensure(
    lambda result, descriptor_names: result.shape
    == ((len(DEFAULT_DESCRIPTOR_NAMES) if descriptor_names is None else len(descriptor_names)),),
    "result length must match the requested descriptor list",
)
def compute_descriptors(
    mol: object,
    descriptor_names: Sequence[str] | None = None,
) -> NDArray[np.float64]:
    """Return selected RDKit molecular descriptors as a float vector.

    The default descriptor list is a compact QSAR feature set covering size,
    polarity, hydrogen bonding, ring structure, topology, charge, and Morgan
    density. Custom descriptor names are resolved against rdkit.Chem.Descriptors.
    """
    descriptors = _load_rdkit_module("rdkit.Chem.Descriptors")
    names = _descriptor_tuple(descriptor_names)
    values: list[float] = []
    for name in names:
        descriptor_fn = getattr(descriptors, name, None)
        if not callable(descriptor_fn):
            raise ValueError(f"unknown RDKit descriptor: {name!r}")
        values.append(float(descriptor_fn(mol)))

    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("RDKit descriptor calculation produced a non-finite value")
    return result

@register_atom(witness_morgan_fingerprint)
@icontract.require(lambda mol: _has_valid_mol_surface(mol), "mol must be an RDKit molecule")
@icontract.require(lambda radius: radius >= 0, "radius must be non-negative")
@icontract.require(lambda n_bits: n_bits > 0, "n_bits must be positive")
@icontract.ensure(lambda result, n_bits: result.shape == (n_bits,), "result length must equal n_bits")
@icontract.ensure(lambda result: np.all((result == 0) | (result == 1)), "fingerprint must be binary")
def morgan_fingerprint(mol: object, radius: int = 2, n_bits: int = 1024) -> NDArray[np.int_]:
    """Compute a dense binary Morgan circular fingerprint vector.

    RDKit returns a compact bit-vector object. This atom converts it to a NumPy
    integer array so downstream model code can concatenate it with other
    numeric features without RDKit-specific adapters.
    """
    rd_mol_descriptors = _load_rdkit_module("rdkit.Chem.rdMolDescriptors")
    data_structs = _load_rdkit_module("rdkit.DataStructs")
    bit_vector = rd_mol_descriptors.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits,
    )
    result = np.zeros((n_bits,), dtype=np.int_)
    data_structs.ConvertToNumpyArray(bit_vector, result)
    return result

@register_atom(witness_maccs_keys)
@icontract.require(lambda mol: _has_valid_mol_surface(mol), "mol must be an RDKit molecule")
@icontract.ensure(lambda result: result.shape == (166,), "result must contain the 166 public MACCS keys")
@icontract.ensure(lambda result: np.all((result == 0) | (result == 1)), "MACCS keys must be binary")
def maccs_keys(mol: object) -> NDArray[np.int_]:
    """Compute the public 166-bit MACCS structural key vector.

    RDKit stores MACCS keys in a 167-bit vector with bit zero unused. This atom
    drops that padding bit and returns the 166 public structural keys.
    """
    maccs_module = _load_rdkit_module("rdkit.Chem.MACCSkeys")
    data_structs = _load_rdkit_module("rdkit.DataStructs")
    bit_vector = maccs_module.GenMACCSKeys(mol)
    padded = np.zeros((167,), dtype=np.int_)
    data_structs.ConvertToNumpyArray(bit_vector, padded)
    return padded[1:]

from __future__ import annotations

"""Ghost witnesses for chemical descriptor and fingerprint atoms."""

from sciona.ghost.abstract import AbstractArray, AbstractScalar


def witness_smiles_to_mol(smiles: AbstractScalar) -> AbstractScalar:
    """Model SMILES parsing as producing one molecule graph handle."""
    return AbstractScalar(dtype="object", min_val=1, max_val=1)


def witness_compute_descriptors(
    mol: AbstractScalar,
    descriptor_names: AbstractScalar | None = None,
) -> AbstractArray:
    """Model descriptor extraction as producing the default 20-value vector."""
    return AbstractArray(shape=(20,), dtype="float64")


def witness_morgan_fingerprint(
    mol: AbstractScalar,
    radius: AbstractScalar | None = None,
    n_bits: AbstractScalar | None = None,
) -> AbstractArray:
    """Model Morgan fingerprinting as producing a binary vector."""
    length = 1024
    if n_bits is not None and n_bits.max_val is not None and n_bits.max_val > 0:
        length = int(n_bits.max_val)
    return AbstractArray(shape=(length,), dtype="int64", min_val=0, max_val=1)


def witness_maccs_keys(mol: AbstractScalar) -> AbstractArray:
    """Model MACCS key extraction as producing the 166 public keys."""
    return AbstractArray(shape=(166,), dtype="int64", min_val=0, max_val=1)


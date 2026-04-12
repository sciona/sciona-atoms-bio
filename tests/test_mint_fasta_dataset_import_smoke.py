import importlib


def test_mint_fasta_dataset_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.bio.mint.fasta_dataset") is not None
    assert importlib.import_module("sciona.probes.bio.mint_fasta_dataset") is not None

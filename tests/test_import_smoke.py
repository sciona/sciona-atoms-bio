import importlib

def test_bio_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.bio.alphafold") is not None
    assert importlib.import_module("sciona.probes.bio.alphafold") is not None

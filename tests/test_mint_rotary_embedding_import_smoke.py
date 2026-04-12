import importlib


def test_mint_rotary_embedding_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.bio.mint.rotary_embedding") is not None
    assert importlib.import_module("sciona.probes.bio.mint_rotary_embedding") is not None

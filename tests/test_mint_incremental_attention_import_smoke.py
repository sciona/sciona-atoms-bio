import importlib


def test_mint_incremental_attention_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.bio.mint.incremental_attention") is not None
    assert importlib.import_module("sciona.probes.bio.mint_incremental_attention") is not None

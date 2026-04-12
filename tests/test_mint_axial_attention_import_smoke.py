import importlib


def test_mint_axial_attention_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.bio.mint.axial_attention") is not None
    assert importlib.import_module("sciona.probes.bio.mint_axial_attention") is not None

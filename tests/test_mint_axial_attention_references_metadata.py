from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "src" / "sciona" / "atoms" / "bio" / "mint"
REFERENCES = ROOT / "references.json"
REGISTRY = REPO_ROOT / "data" / "references" / "registry.json"

TARGET_FQDNS = {
    "sciona.atoms.bio.mint.axial_attention.row_self_attention",
    "sciona.atoms.bio.mint.axial_attention.rowselfattention",
}


def test_mint_axial_attention_references_bind_to_local_registry() -> None:
    refs = json.loads(REFERENCES.read_text())
    registry = json.loads(REGISTRY.read_text())["references"]
    atom_entries = refs["atoms"]

    keys_by_fqdn = {
        key.split("@", 1)[0]: value
        for key, value in atom_entries.items()
        if key.split("@", 1)[0] in TARGET_FQDNS
    }
    assert set(keys_by_fqdn) == TARGET_FQDNS

    for fqdn, entry in keys_by_fqdn.items():
        assert entry["references"]
        assert "axial_attention.py" in next(
            key for key in atom_entries if key.startswith(fqdn + "@")
        )
        for reference in entry["references"]:
            assert reference["ref_id"] in registry
            metadata = reference["match_metadata"]
            assert metadata["match_type"] == "manual_direct_audit"
            assert metadata["confidence"] in {"high", "medium"}
            assert metadata["notes"]
            assert "mint/axial_attention.py:12" in metadata["matched_nodes"]

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "licenses" / "provider_license.json"


def test_bio_license_manifest_is_repo_defaulted_and_slice_scoped() -> None:
    data = json.loads(MANIFEST.read_text())

    assert data["provider_repo"] == "sciona-atoms-bio"
    assert data["repository_default"]["license_expression"] == "NOASSERTION"
    assert data["repository_default"]["status"] == "unknown"
    assert data["family_overrides"] == []
    assert data["scope"] == ["bio.molecular_docking"]

    families = [entry["family"] for entry in data["family_inventory"]]
    assert families == ["bio.molecular_docking"]

    entry = data["family_inventory"][0]
    assert entry["license_expression"] == "NOASSERTION"
    assert entry["status"] == "unknown"
    assert entry["authoritative_sources"] == ["src/sciona/atoms/bio/molecular_docking"]

    unresolved = {item["family"]: item["status"] for item in data["unresolved_families"]}
    assert unresolved == {
        "bio.alphafold": "unknown",
        "bio.hpdb": "unknown",
        "bio.mint": "unknown",
    }

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "data" / "review_bundles" / "medical_imaging_3d_augmentation.review_bundle.json"

AUGMENTATION_FQDNS = {
    "sciona.atoms.medical_imaging_3d.augmentation.add_gaussian_noise_3d",
    "sciona.atoms.medical_imaging_3d.augmentation.elastic_deform_3d",
    "sciona.atoms.medical_imaging_3d.augmentation.random_rotate_3d",
    "sciona.atoms.medical_imaging_3d.augmentation.scale_volume_3d",
}


def _bundle() -> dict:
    return json.loads(BUNDLE.read_text())


def test_medical_imaging_3d_augmentation_review_bundle_is_catalog_ready() -> None:
    bundle = _bundle()

    assert bundle["schema_version"] == "1.0"
    assert bundle["bundle_id"] == "sciona.atoms.review_bundle.medical_imaging_3d.augmentation.v1"
    assert bundle["provider_repo"] == "sciona-atoms-bio"
    assert bundle["family"] == "medical_imaging_3d.augmentation"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "catalog_ready"
    assert bundle["blocking_findings"] == []
    assert bundle["required_actions"] == []

    for rel in bundle["authoritative_sources"]:
        assert (REPO_ROOT / rel).exists()


def test_medical_imaging_3d_augmentation_review_bundle_points_to_live_functions() -> None:
    rows = {row["atom_name"]: row for row in _bundle()["rows"]}
    assert set(rows) == AUGMENTATION_FQDNS

    for fqdn, row in rows.items():
        assert row["atom_key"] == fqdn
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] == "pass_with_limits"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["blocking_findings"] == []
        assert row["required_actions"] == []
        assert row["source_paths"]

        source_rel, _, line_text = row["source_path"].partition(":")
        assert line_text
        source_file = REPO_ROOT / "src" / source_rel
        assert source_file.exists()

        module_name, function_name = fqdn.rsplit(".", 1)
        function = getattr(importlib.import_module(module_name), function_name)
        source_text, _ = inspect.getsourcelines(function)
        assert f"def {function_name}(" in "".join(source_text)
        assert source_file.read_text().splitlines()[int(line_text) - 1].lstrip().startswith(
            f"def {function_name}("
        )

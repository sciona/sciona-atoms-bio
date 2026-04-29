from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "src" / "sciona" / "atoms" / "medical_imaging_3d" / "preprocessing"
REGISTRY = REPO_ROOT / "data" / "references" / "registry.json"

PREPROCESSING_FQDNS = {
    "sciona.atoms.medical_imaging_3d.preprocessing.crop_to_mask",
    "sciona.atoms.medical_imaging_3d.preprocessing.dicom_to_hounsfield",
    "sciona.atoms.medical_imaging_3d.preprocessing.dicom_window",
    "sciona.atoms.medical_imaging_3d.preprocessing.extract_25d_slices",
    "sciona.atoms.medical_imaging_3d.preprocessing.filter_small_components",
    "sciona.atoms.medical_imaging_3d.preprocessing.macenko_normalize",
    "sciona.atoms.medical_imaging_3d.preprocessing.macenko_stain_vectors",
    "sciona.atoms.medical_imaging_3d.preprocessing.max_intensity_projection",
    "sciona.atoms.medical_imaging_3d.preprocessing.parse_a3d_volume",
    "sciona.atoms.medical_imaging_3d.preprocessing.resample_volume",
}


def test_medical_imaging_3d_preprocessing_references_bind_to_local_registry() -> None:
    refs = json.loads((ROOT / "references.json").read_text())
    registry = json.loads(REGISTRY.read_text())["references"]

    assert refs["schema_version"] == "1.1"
    assert {key.split("@", 1)[0] for key in refs["atoms"]} == PREPROCESSING_FQDNS

    for atom_key, record in refs["atoms"].items():
        location = atom_key.split("@", 1)[1]
        source_rel, _, line_text = location.partition(":")
        source_file = REPO_ROOT / "src" / source_rel
        function_name = atom_key.split("@", 1)[0].rsplit(".", 1)[-1]

        assert source_file.exists()
        assert line_text.isdigit()
        assert f"def {function_name}(" in source_file.read_text().splitlines()[int(line_text) - 1]
        assert record["references"]

        for reference in record["references"]:
            assert reference["ref_id"] in registry
            metadata = reference["match_metadata"]
            assert metadata["match_type"]
            assert metadata["confidence"] in {"low", "medium", "high"}
            assert metadata["notes"]

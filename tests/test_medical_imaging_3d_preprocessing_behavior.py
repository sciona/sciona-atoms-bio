from __future__ import annotations

import numpy as np
import pytest

from sciona.atoms.medical_imaging_3d.preprocessing import (
    crop_to_mask,
    dicom_to_hounsfield,
    dicom_window,
    extract_25d_slices,
    filter_small_components,
    macenko_normalize,
    macenko_stain_vectors,
    max_intensity_projection,
    parse_a3d_volume,
    resample_volume,
)


def _histology_patch() -> np.ndarray:
    return np.array(
        [
            [[180, 80, 160], [200, 120, 170], [230, 180, 210]],
            [[100, 40, 140], [210, 90, 120], [240, 200, 220]],
            [[120, 55, 150], [190, 105, 165], [225, 170, 205]],
        ],
        dtype=np.uint8,
    )


def test_dicom_hounsfield_and_exact_window_match_expected_values() -> None:
    pixels = np.array([-100, 0, 50, 100, 200], dtype=np.int16)

    hu = dicom_to_hounsfield(pixels, slope=2.0, intercept=-1024.0)
    np.testing.assert_allclose(hu, np.array([-1224.0, -1024.0, -924.0, -824.0, -624.0]))

    windowed = dicom_window(pixels, window_center=50.0, window_width=100.0, exact=True)
    np.testing.assert_allclose(windowed, np.array([0.0, 0.0, 127.5, 255.0, 255.0]))


def test_resample_volume_preserves_constant_volume() -> None:
    volume = np.ones((2, 2, 2), dtype=np.float64)

    result = resample_volume(volume, (1.0, 1.0, 1.0), (0.5, 0.5, 0.5))

    assert result.shape == (4, 4, 4)
    np.testing.assert_allclose(result, np.ones((4, 4, 4)))


def test_macenko_stain_vectors_and_normalization_have_expected_shapes() -> None:
    image = _histology_patch()

    stain_matrix, concentrations = macenko_stain_vectors(image, alpha=5.0, beta=0.05)
    normalized = macenko_normalize(image, stain_matrix, alpha=5.0, beta=0.05)

    assert stain_matrix.shape == (3, 2)
    assert concentrations.shape == (image.shape[0] * image.shape[1], 2)
    assert np.all(np.isfinite(stain_matrix))
    assert np.all(concentrations >= 0.0)
    assert normalized.shape == image.shape
    assert normalized.dtype == np.uint8


def test_max_intensity_projection_collapses_requested_axis() -> None:
    volume = np.zeros((3, 4, 5), dtype=np.float64)
    volume[2, 1, 3] = 7.0

    result = max_intensity_projection(volume, angle=0.0, projection_axis=0, order=0)

    assert result.shape == (4, 5)
    assert result[1, 3] == 7.0


def test_extract_25d_slices_uses_edge_padding_at_volume_boundary() -> None:
    volume = np.arange(4 * 2 * 2, dtype=np.float64).reshape(4, 2, 2)

    result = extract_25d_slices(volume, center_idx=0, num_adjacent=2)

    assert result.shape == (5, 2, 2)
    np.testing.assert_array_equal(result[0], volume[0])
    np.testing.assert_array_equal(result[1], volume[0])
    np.testing.assert_array_equal(result[2], volume[0])
    np.testing.assert_array_equal(result[3], volume[1])
    np.testing.assert_array_equal(result[4], volume[2])


def test_crop_to_mask_returns_volume_and_half_open_bounds() -> None:
    volume = np.arange(10 * 10 * 10).reshape(10, 10, 10)
    mask = np.zeros_like(volume, dtype=bool)
    mask[5, 5, 5] = True

    cropped, bounds = crop_to_mask(volume, mask, margin=1)

    assert cropped.shape == (3, 3, 3)
    assert bounds == ((4, 7), (4, 7), (4, 7))
    assert cropped[1, 1, 1] == volume[5, 5, 5]


def test_filter_small_components_removes_isolated_binary_noise() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    mask[8, 8] = 1

    result = filter_small_components(mask, min_size=5)

    assert result.dtype == np.bool_
    assert result[1:4, 1:4].all()
    assert not result[8, 8]


def test_parse_a3d_volume_transposes_yzx_bytes_to_zyx_float_volume() -> None:
    payload = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)

    result = parse_a3d_volume(payload.tobytes(), data_scale_factor=0.5, shape_yzx=(2, 3, 4))

    assert result.shape == (3, 2, 4)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.transpose(payload, (1, 0, 2)).astype(np.float32) * 0.5)


def test_empty_mask_is_rejected() -> None:
    with pytest.raises(Exception):
        crop_to_mask(np.zeros((2, 2, 2)), np.zeros((2, 2, 2), dtype=bool))

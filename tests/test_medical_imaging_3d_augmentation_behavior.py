from __future__ import annotations

import numpy as np
import pytest

from sciona.atoms.medical_imaging_3d.augmentation import (
    add_gaussian_noise_3d,
    elastic_deform_3d,
    random_rotate_3d,
    scale_volume_3d,
)


def test_random_rotate_3d_preserves_shape_and_supports_leading_channels() -> None:
    volume = np.zeros((2, 8, 8, 8), dtype=np.float32)
    volume[0, 4, 4, 7] = 1.0
    volume[1, 4, 4, 7] = 2.0

    result = random_rotate_3d(volume, angles_deg=(0.0, 90.0, 0.0), order=0)

    assert result.shape == volume.shape
    assert result.dtype == np.float64
    assert np.unravel_index(np.argmax(result[0]), result[0].shape) == (7, 4, 3)
    assert result[1, 7, 4, 3] == 2.0


def test_elastic_deform_3d_is_seeded_and_shape_preserving() -> None:
    volume = np.zeros((8, 8, 8), dtype=np.float64)
    volume[2:6, 2:6, 2:6] = 1.0

    first = elastic_deform_3d(volume, sigma=3.0, alpha=1.5, seed=17, order=1)
    second = elastic_deform_3d(volume, sigma=3.0, alpha=1.5, seed=17, order=1)
    no_warp = elastic_deform_3d(volume, sigma=3.0, alpha=0.0, seed=17, order=1)

    assert first.shape == volume.shape
    assert np.all(np.isfinite(first))
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(no_warp, volume)


def test_scale_volume_3d_recenters_zoomed_volume() -> None:
    shell = np.zeros((8, 8, 8), dtype=np.float64)
    shell[[0, -1], :, :] = 1.0
    shell[:, [0, -1], :] = 1.0
    shell[:, :, [0, -1]] = 1.0

    result = scale_volume_3d(shell, scale_factors=(0.5, 0.5, 0.5), order=0)

    assert result.shape == shell.shape
    assert np.count_nonzero(result[:2]) == 0
    assert np.count_nonzero(result[-2:]) == 0
    assert np.count_nonzero(result[:, :2]) == 0
    assert np.count_nonzero(result[:, -2:]) == 0
    assert np.count_nonzero(result[:, :, :2]) == 0
    assert np.count_nonzero(result[:, :, -2:]) == 0
    assert np.count_nonzero(result[2:6, 2:6, 2:6]) > 0


def test_add_gaussian_noise_3d_is_reproducible_and_preserves_dtype() -> None:
    volume = np.zeros((24, 24, 24), dtype=np.float32)

    first = add_gaussian_noise_3d(volume, mean=0.0, std=0.2, seed=11)
    second = add_gaussian_noise_3d(volume, mean=0.0, std=0.2, seed=11)

    assert first.shape == volume.shape
    assert first.dtype == volume.dtype
    np.testing.assert_allclose(first, second)
    assert abs(float(np.mean(first))) < 0.02
    assert 0.17 < float(np.std(first)) < 0.23


def test_invalid_augmentation_inputs_are_rejected() -> None:
    with pytest.raises(Exception):
        random_rotate_3d(np.zeros((8, 8)), angles_deg=(0.0, 0.0, 0.0))

    with pytest.raises(Exception):
        scale_volume_3d(np.zeros((8, 8, 8)), scale_factors=(1.0, 0.0, 1.0))

    with pytest.raises(Exception):
        add_gaussian_noise_3d(np.zeros((8, 8, 8)), mean=0.0, std=-1.0, seed=1)

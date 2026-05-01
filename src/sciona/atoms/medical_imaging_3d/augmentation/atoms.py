from __future__ import annotations

"""Pure 3D augmentation atoms for medical and tomographic volumes."""

from collections.abc import Callable, Sequence

import icontract
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from sciona.atoms.medical_imaging_3d.augmentation.witnesses import (
    witness_add_gaussian_noise_3d,
    witness_elastic_deform_3d,
    witness_random_rotate_3d,
    witness_scale_volume_3d,
)
from sciona.ghost.registry import register_atom


Angles3 = tuple[float, float, float]
ScaleFactors3 = tuple[float, float, float]


def _is_real_numeric_array(value: np.ndarray) -> bool:
    return np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.floating)


def _has_spatial_volume(value: np.ndarray) -> bool:
    return value.ndim >= 3 and all(size > 0 for size in value.shape[-3:])


def _finite_sequence(values: Sequence[float], length: int) -> bool:
    return len(values) == length and all(np.isfinite(item) for item in values)


def _positive_sequence(values: Sequence[float], length: int) -> bool:
    return _finite_sequence(values, length) and all(item > 0.0 for item in values)


def _all_finite(value: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(value)))


def _spatial_center(shape: tuple[int, int, int]) -> np.ndarray:
    return (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0


def _apply_last3(source: np.ndarray, transform: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    leading_shape = source.shape[:-3]
    spatial_shape = source.shape[-3:]
    if not leading_shape:
        return transform(source)

    output = np.empty(source.shape, dtype=np.float64)
    flat_source = source.reshape((-1, *spatial_shape))
    flat_output = output.reshape((-1, *spatial_shape))
    for index, channel in enumerate(flat_source):
        flat_output[index] = transform(channel)
    return output


def _affine_rotate_single(volume: np.ndarray, angles_deg: Angles3, order: int) -> np.ndarray:
    rotation = Rotation.from_euler("xyz", angles_deg, degrees=True)
    matrix = rotation.inv().as_matrix()
    center = _spatial_center(volume.shape)
    offset = center - matrix @ center
    return ndimage.affine_transform(
        volume,
        matrix=matrix,
        offset=offset,
        output_shape=volume.shape,
        order=order,
        mode="nearest",
        prefilter=order > 1,
    )


def _center_crop_or_pad(source: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    result = source
    crop_slices: list[slice] = []
    for axis, target in enumerate(target_shape):
        current = result.shape[axis]
        if current > target:
            start = (current - target) // 2
            crop_slices.append(slice(start, start + target))
        else:
            crop_slices.append(slice(None))
    result = result[tuple(crop_slices)]

    pad_width: list[tuple[int, int]] = []
    for axis, target in enumerate(target_shape):
        current = result.shape[axis]
        missing = max(0, target - current)
        before = missing // 2
        pad_width.append((before, missing - before))
    if any(before or after for before, after in pad_width):
        result = np.pad(result, pad_width, mode="constant", constant_values=0.0)
    return result


def _scale_single(volume: np.ndarray, scale_factors: ScaleFactors3, order: int) -> np.ndarray:
    zoomed = ndimage.zoom(
        volume,
        zoom=scale_factors,
        order=order,
        mode="nearest",
        prefilter=order > 1,
    )
    return _center_crop_or_pad(zoomed, volume.shape)


def _displacement_field(
    shape: tuple[int, int, int],
    sigma: float,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fields = []
    for _ in range(3):
        raw = rng.uniform(-1.0, 1.0, size=shape)
        smooth = ndimage.gaussian_filter(raw, sigma=float(sigma), mode="reflect")
        fields.append(smooth * float(alpha))
    return fields[0], fields[1], fields[2]


def _elastic_single(
    volume: np.ndarray,
    displacements: tuple[np.ndarray, np.ndarray, np.ndarray],
    order: int,
) -> np.ndarray:
    grid = np.meshgrid(
        np.arange(volume.shape[0], dtype=np.float64),
        np.arange(volume.shape[1], dtype=np.float64),
        np.arange(volume.shape[2], dtype=np.float64),
        indexing="ij",
    )
    coordinates = [axis_grid + offset for axis_grid, offset in zip(grid, displacements, strict=True)]
    return ndimage.map_coordinates(
        volume,
        coordinates,
        order=order,
        mode="nearest",
        prefilter=order > 1,
    )


def _cast_noise_like(noisy: np.ndarray, template: np.ndarray) -> np.ndarray:
    if np.issubdtype(template.dtype, np.floating):
        return noisy.astype(template.dtype, copy=False)
    info = np.iinfo(template.dtype)
    return np.clip(np.rint(noisy), info.min, info.max).astype(template.dtype, copy=False)


@register_atom(witness_random_rotate_3d)
@icontract.require(lambda volume: isinstance(volume, np.ndarray), "volume must be a NumPy array")
@icontract.require(lambda volume: _has_spatial_volume(volume), "volume must have non-empty last three spatial axes")
@icontract.require(lambda volume: _is_real_numeric_array(volume), "volume must be a real numeric array")
@icontract.require(lambda volume: _all_finite(volume), "volume values must be finite")
@icontract.require(lambda angles_deg: _finite_sequence(angles_deg, 3), "angles_deg must contain three finite values")
@icontract.require(lambda order: order in {0, 1, 2, 3, 4, 5}, "order must be a scipy spline order")
@icontract.ensure(lambda volume, result: result.shape == volume.shape, "result preserves input shape")
@icontract.ensure(lambda result: _all_finite(result), "result must be finite")
def random_rotate_3d(volume: np.ndarray, angles_deg: Angles3, order: int = 1) -> np.ndarray:
    """Rotate a volume around its geometric center using one inverse 3D affine resampling pass."""
    source = np.asarray(volume, dtype=np.float64)
    angles = (float(angles_deg[0]), float(angles_deg[1]), float(angles_deg[2]))
    return _apply_last3(source, lambda channel: _affine_rotate_single(channel, angles, int(order)))


@register_atom(witness_elastic_deform_3d)
@icontract.require(lambda volume: isinstance(volume, np.ndarray), "volume must be a NumPy array")
@icontract.require(lambda volume: _has_spatial_volume(volume), "volume must have non-empty last three spatial axes")
@icontract.require(lambda volume: _is_real_numeric_array(volume), "volume must be a real numeric array")
@icontract.require(lambda volume: _all_finite(volume), "volume values must be finite")
@icontract.require(lambda sigma: np.isfinite(sigma) and sigma > 0.0, "sigma must be positive")
@icontract.require(lambda alpha: np.isfinite(alpha) and alpha >= 0.0, "alpha must be non-negative")
@icontract.require(lambda seed: isinstance(seed, int), "seed must be an integer")
@icontract.require(lambda order: order in {0, 1, 2, 3, 4, 5}, "order must be a scipy spline order")
@icontract.ensure(lambda volume, result: result.shape == volume.shape, "result preserves input shape")
@icontract.ensure(lambda result: _all_finite(result), "result must be finite")
def elastic_deform_3d(
    volume: np.ndarray,
    sigma: float,
    alpha: float,
    seed: int,
    order: int = 1,
) -> np.ndarray:
    """Warp a volume with a seeded, smoothed 3D displacement field over the last three axes."""
    source = np.asarray(volume, dtype=np.float64)
    if float(alpha) == 0.0:
        return source.copy()
    displacements = _displacement_field(source.shape[-3:], float(sigma), float(alpha), int(seed))
    return _apply_last3(source, lambda channel: _elastic_single(channel, displacements, int(order)))


@register_atom(witness_scale_volume_3d)
@icontract.require(lambda volume: isinstance(volume, np.ndarray), "volume must be a NumPy array")
@icontract.require(lambda volume: _has_spatial_volume(volume), "volume must have non-empty last three spatial axes")
@icontract.require(lambda volume: _is_real_numeric_array(volume), "volume must be a real numeric array")
@icontract.require(lambda volume: _all_finite(volume), "volume values must be finite")
@icontract.require(lambda scale_factors: _positive_sequence(scale_factors, 3), "scale_factors must contain three positive values")
@icontract.require(lambda order: order in {0, 1, 2, 3, 4, 5}, "order must be a scipy spline order")
@icontract.ensure(lambda volume, result: result.shape == volume.shape, "result preserves input shape")
@icontract.ensure(lambda result: _all_finite(result), "result must be finite")
def scale_volume_3d(volume: np.ndarray, scale_factors: ScaleFactors3, order: int = 1) -> np.ndarray:
    """Zoom the last three volume axes, then center-crop or zero-pad back to the input shape."""
    source = np.asarray(volume, dtype=np.float64)
    factors = (float(scale_factors[0]), float(scale_factors[1]), float(scale_factors[2]))
    return _apply_last3(source, lambda channel: _scale_single(channel, factors, int(order)))


@register_atom(witness_add_gaussian_noise_3d)
@icontract.require(lambda volume: isinstance(volume, np.ndarray), "volume must be a NumPy array")
@icontract.require(lambda volume: _has_spatial_volume(volume), "volume must have non-empty last three spatial axes")
@icontract.require(lambda volume: _is_real_numeric_array(volume), "volume must be a real numeric array")
@icontract.require(lambda volume: _all_finite(volume), "volume values must be finite")
@icontract.require(lambda mean: np.isfinite(mean), "mean must be finite")
@icontract.require(lambda std: np.isfinite(std) and std >= 0.0, "std must be non-negative")
@icontract.require(lambda seed: isinstance(seed, int), "seed must be an integer")
@icontract.ensure(lambda volume, result: result.shape == volume.shape, "result preserves input shape")
@icontract.ensure(lambda volume, result: result.dtype == volume.dtype, "result preserves input dtype")
@icontract.ensure(lambda result: _all_finite(result), "result must be finite")
def add_gaussian_noise_3d(volume: np.ndarray, mean: float, std: float, seed: int) -> np.ndarray:
    """Add deterministic Gaussian intensity noise from an isolated NumPy random generator."""
    source = np.asarray(volume)
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(float(mean), float(std), size=source.shape)
    return _cast_noise_like(source.astype(np.float64) + noise, source)

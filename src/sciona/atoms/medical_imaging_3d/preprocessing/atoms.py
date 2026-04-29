from __future__ import annotations

"""Pure preprocessing atoms for medical-imaging arrays."""

from collections.abc import Sequence

import icontract
import numpy as np
from scipy import ndimage

from sciona.atoms.medical_imaging_3d.preprocessing.witnesses import (
    witness_crop_to_mask,
    witness_dicom_to_hounsfield,
    witness_dicom_window,
    witness_extract_25d_slices,
    witness_filter_small_components,
    witness_macenko_normalize,
    witness_macenko_stain_vectors,
    witness_max_intensity_projection,
    witness_parse_a3d_volume,
    witness_resample_volume,
)
from sciona.ghost.registry import register_atom


Spacing3 = tuple[float, float, float]
Shape3 = tuple[int, int, int]
SliceBounds3 = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


def _is_numeric_array(value: np.ndarray) -> bool:
    return np.issubdtype(value.dtype, np.number)


def _is_binary_array(value: np.ndarray) -> bool:
    if value.dtype == np.bool_:
        return True
    return bool(np.all((value == 0) | (value == 1)))


def _spacing_is_valid(spacing: Sequence[float]) -> bool:
    return len(spacing) == 3 and all(np.isfinite(item) and item > 0.0 for item in spacing)


def _target_shape(volume_shape: tuple[int, ...], current_spacing: Spacing3, target_spacing: Spacing3) -> Shape3:
    shape = np.asarray(volume_shape, dtype=np.float64)
    current = np.asarray(current_spacing, dtype=np.float64)
    target = np.asarray(target_spacing, dtype=np.float64)
    return tuple(np.maximum(1, np.round(shape * current / target).astype(np.int64)).tolist())  # type: ignore[return-value]


def _as_rgb_float(image: np.ndarray, luminosity: float) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float64)
    if rgb.max(initial=0.0) <= 1.0:
        rgb = rgb * luminosity
    return np.clip(rgb, 0.0, luminosity)


def _optical_density(image: np.ndarray, luminosity: float) -> np.ndarray:
    rgb = _as_rgb_float(image, luminosity)
    return -np.log((rgb + 1.0) / luminosity)


def _macenko_estimate(image: np.ndarray, alpha: float, beta: float, luminosity: float) -> tuple[np.ndarray, np.ndarray]:
    od = _optical_density(image, luminosity).reshape(-1, 3)
    tissue = od[np.any(od > beta, axis=1)]
    if tissue.shape[0] < 2:
        raise ValueError("image must contain at least two tissue-like RGB pixels")

    _, _, vt = np.linalg.svd(tissue, full_matrices=False)
    plane = vt[:2].T
    projected = tissue @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    lo = np.percentile(angles, alpha)
    hi = np.percentile(angles, 100.0 - alpha)

    vectors = np.stack(
        [
            plane @ np.array([np.cos(lo), np.sin(lo)], dtype=np.float64),
            plane @ np.array([np.cos(hi), np.sin(hi)], dtype=np.float64),
        ],
        axis=1,
    )
    vectors = np.abs(vectors)
    norms = np.linalg.norm(vectors, axis=0)
    if bool(np.any(norms <= 0.0)):
        raise ValueError("stain vector estimation produced a zero vector")
    stain_matrix = vectors / norms

    concentrations = np.linalg.lstsq(stain_matrix, od.T, rcond=None)[0].T
    return stain_matrix, np.clip(concentrations, 0.0, None)


@register_atom(witness_dicom_to_hounsfield)
@icontract.require(lambda pixel_array: isinstance(pixel_array, np.ndarray), "pixel_array must be a NumPy array")
@icontract.require(lambda pixel_array: _is_numeric_array(pixel_array), "pixel_array must be numeric")
@icontract.require(lambda slope: np.isfinite(slope), "slope must be finite")
@icontract.require(lambda intercept: np.isfinite(intercept), "intercept must be finite")
@icontract.ensure(lambda pixel_array, result: result.shape == pixel_array.shape, "result preserves input shape")
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))), "result must be finite")
def dicom_to_hounsfield(pixel_array: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Apply DICOM rescale slope and intercept to produce physical CT intensity values."""
    pixels = np.asarray(pixel_array, dtype=np.float64)
    return pixels * float(slope) + float(intercept)


@register_atom(witness_dicom_window)
@icontract.require(lambda pixel_array: isinstance(pixel_array, np.ndarray), "pixel_array must be a NumPy array")
@icontract.require(lambda pixel_array: _is_numeric_array(pixel_array), "pixel_array must be numeric")
@icontract.require(lambda pixel_array: pixel_array.ndim in {1, 2, 3}, "pixel_array must be grayscale planar or volumetric")
@icontract.require(lambda window_center: np.isfinite(window_center), "window_center must be finite")
@icontract.require(lambda window_width: np.isfinite(window_width) and window_width > 0.0, "window_width must be positive")
@icontract.require(lambda y_min, y_max: np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min, "output range must increase")
@icontract.ensure(lambda pixel_array, result: result.shape == pixel_array.shape, "result preserves input shape")
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))), "result must be finite")
@icontract.ensure(lambda y_min, y_max, result: bool(np.all((result >= y_min) & (result <= y_max))), "result is clipped to range")
def dicom_window(
    pixel_array: np.ndarray,
    window_center: float,
    window_width: float,
    exact: bool = False,
    y_min: float = 0.0,
    y_max: float = 255.0,
) -> np.ndarray:
    """Map grayscale DICOM intensities into a bounded display range using VOI windowing."""
    pixels = np.nan_to_num(np.asarray(pixel_array, dtype=np.float64), copy=False)
    center = float(window_center)
    width = float(window_width)
    output_min = float(y_min)
    output_max = float(y_max)

    if exact:
        lower = center - width / 2.0
        upper = center + width / 2.0
        scaled = (pixels - lower) / width
    elif width == 1.0:
        return np.where(pixels <= center - 0.5, output_min, output_max).astype(np.float64)
    else:
        lower = center - 0.5 - (width - 1.0) / 2.0
        upper = center - 0.5 + (width - 1.0) / 2.0
        scaled = ((pixels - (center - 0.5)) / (width - 1.0)) + 0.5

    clipped = np.clip(scaled, 0.0, 1.0)
    clipped = np.where(pixels <= lower, 0.0, clipped)
    clipped = np.where(pixels > upper, 1.0, clipped)
    return clipped * (output_max - output_min) + output_min


@register_atom(witness_resample_volume)
@icontract.require(lambda volume: isinstance(volume, np.ndarray), "volume must be a NumPy array")
@icontract.require(lambda volume: volume.ndim == 3, "volume must be three-dimensional")
@icontract.require(lambda volume: _is_numeric_array(volume), "volume must be numeric")
@icontract.require(lambda current_spacing: _spacing_is_valid(current_spacing), "current_spacing must contain three positive values")
@icontract.require(lambda target_spacing: _spacing_is_valid(target_spacing), "target_spacing must contain three positive values")
@icontract.require(lambda order: order in {0, 1, 2, 3, 4, 5}, "order must be a scipy spline order")
@icontract.ensure(
    lambda volume, current_spacing, target_spacing, result: result.shape
    == _target_shape(volume.shape, current_spacing, target_spacing),
    "result shape follows physical spacing ratio",
)
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))), "result must be finite")
def resample_volume(
    volume: np.ndarray,
    current_spacing: Spacing3,
    target_spacing: Spacing3,
    order: int = 1,
) -> np.ndarray:
    """Resample a 3D medical volume to a requested physical voxel spacing."""
    source = np.asarray(volume, dtype=np.float64)
    new_shape = _target_shape(source.shape, current_spacing, target_spacing)
    coordinates = np.meshgrid(
        *[np.linspace(0.0, source.shape[axis] - 1.0, new_shape[axis]) for axis in range(3)],
        indexing="ij",
    )
    return ndimage.map_coordinates(source, coordinates, order=order, mode="nearest")


@register_atom(witness_macenko_stain_vectors)
@icontract.require(lambda image: isinstance(image, np.ndarray), "image must be a NumPy array")
@icontract.require(lambda image: image.ndim == 3 and image.shape[-1] == 3, "image must be RGB")
@icontract.require(lambda image: _is_numeric_array(image), "image must be numeric")
@icontract.require(lambda alpha: np.isfinite(alpha) and 0.0 <= alpha <= 50.0, "alpha must be a percentile in [0, 50]")
@icontract.require(lambda beta: np.isfinite(beta) and beta > 0.0, "beta must be positive")
@icontract.require(lambda luminosity: np.isfinite(luminosity) and luminosity > 0.0, "luminosity must be positive")
@icontract.ensure(lambda result: result[0].shape == (3, 2), "stain matrix has two RGB stain vectors")
@icontract.ensure(lambda result: result[1].ndim == 2 and result[1].shape[1] == 2, "concentrations have two stain channels")
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result[0])) and np.all(np.isfinite(result[1]))), "result must be finite")
@icontract.ensure(lambda result: bool(np.all(result[1] >= 0.0)), "concentrations must be non-negative")
def macenko_stain_vectors(
    image: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.15,
    luminosity: float = 255.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate H&E stain vectors and per-pixel concentrations with the Macenko SVD method."""
    return _macenko_estimate(image, float(alpha), float(beta), float(luminosity))


@register_atom(witness_macenko_normalize)
@icontract.require(lambda image: isinstance(image, np.ndarray), "image must be a NumPy array")
@icontract.require(lambda image: image.ndim == 3 and image.shape[-1] == 3, "image must be RGB")
@icontract.require(lambda target_stain_matrix: isinstance(target_stain_matrix, np.ndarray), "target_stain_matrix must be a NumPy array")
@icontract.require(lambda target_stain_matrix: target_stain_matrix.shape == (3, 2), "target_stain_matrix must have shape (3, 2)")
@icontract.require(lambda target_stain_matrix: bool(np.all(np.isfinite(target_stain_matrix))), "target_stain_matrix must be finite")
@icontract.require(lambda alpha: np.isfinite(alpha) and 0.0 <= alpha <= 50.0, "alpha must be a percentile in [0, 50]")
@icontract.require(lambda beta: np.isfinite(beta) and beta > 0.0, "beta must be positive")
@icontract.require(lambda luminosity: np.isfinite(luminosity) and luminosity > 0.0, "luminosity must be positive")
@icontract.ensure(lambda image, result: result.shape == image.shape, "normalized image preserves shape")
@icontract.ensure(lambda result: result.dtype == np.uint8, "normalized image is uint8 RGB")
def macenko_normalize(
    image: np.ndarray,
    target_stain_matrix: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.15,
    luminosity: float = 255.0,
) -> np.ndarray:
    """Normalize an RGB histology image to an explicit target Macenko stain matrix."""
    _, concentrations = _macenko_estimate(image, float(alpha), float(beta), float(luminosity))
    target = np.asarray(target_stain_matrix, dtype=np.float64)
    reconstructed_od = concentrations @ target.T
    rgb = float(luminosity) * np.exp(-reconstructed_od)
    return np.clip(rgb.reshape(image.shape), 0.0, 255.0).astype(np.uint8)


@register_atom(witness_max_intensity_projection)
@icontract.require(lambda volume: isinstance(volume, np.ndarray), "volume must be a NumPy array")
@icontract.require(lambda volume: volume.ndim == 3, "volume must be three-dimensional")
@icontract.require(lambda volume: _is_numeric_array(volume), "volume must be numeric")
@icontract.require(lambda angle: np.isfinite(angle), "angle must be finite")
@icontract.require(lambda rotation_axes: len(rotation_axes) == 2 and len(set(rotation_axes)) == 2, "rotation_axes must name two axes")
@icontract.require(lambda rotation_axes: all(axis in {0, 1, 2} for axis in rotation_axes), "rotation_axes must be valid volume axes")
@icontract.require(lambda projection_axis: projection_axis in {0, 1, 2}, "projection_axis must be a valid volume axis")
@icontract.require(lambda order: order in {0, 1, 2, 3, 4, 5}, "order must be a scipy spline order")
@icontract.ensure(lambda result: result.ndim == 2, "projection is two-dimensional")
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))), "result must be finite")
def max_intensity_projection(
    volume: np.ndarray,
    angle: float,
    rotation_axes: tuple[int, int] = (1, 2),
    projection_axis: int = 0,
    order: int = 1,
) -> np.ndarray:
    """Rotate a 3D volume and collapse it into a 2D maximum-intensity projection."""
    rotated = ndimage.rotate(
        np.asarray(volume, dtype=np.float64),
        angle=float(angle),
        axes=rotation_axes,
        reshape=False,
        order=order,
        mode="nearest",
    )
    return np.max(rotated, axis=projection_axis)


@register_atom(witness_extract_25d_slices)
@icontract.require(lambda volume: isinstance(volume, np.ndarray), "volume must be a NumPy array")
@icontract.require(lambda volume: volume.ndim >= 3, "volume must have at least three dimensions")
@icontract.require(lambda center_idx, volume, axis: 0 <= center_idx < volume.shape[axis], "center_idx must be in bounds")
@icontract.require(lambda num_adjacent: num_adjacent >= 0, "num_adjacent must be non-negative")
@icontract.require(lambda axis, volume: -volume.ndim <= axis < volume.ndim, "axis must be valid for volume")
@icontract.ensure(lambda volume, num_adjacent, result: result.shape[0] == 2 * num_adjacent + 1, "result has requested slice count")
@icontract.ensure(lambda volume, axis, result: result.shape[1:] == np.moveaxis(volume, axis, 0).shape[1:], "spatial dimensions are preserved")
def extract_25d_slices(
    volume: np.ndarray,
    center_idx: int,
    num_adjacent: int,
    axis: int = 0,
    mode: str = "edge",
) -> np.ndarray:
    """Stack a central slice and neighboring slices into a channel-first 2.5D tensor."""
    source = np.moveaxis(np.asarray(volume), axis, 0)
    pad_width = [(int(num_adjacent), int(num_adjacent))] + [(0, 0)] * (source.ndim - 1)
    padded = np.pad(source, pad_width, mode=mode)
    start = int(center_idx)
    stop = start + 2 * int(num_adjacent) + 1
    return padded[start:stop]


@register_atom(witness_crop_to_mask)
@icontract.require(lambda volume: isinstance(volume, np.ndarray), "volume must be a NumPy array")
@icontract.require(lambda mask: isinstance(mask, np.ndarray), "mask must be a NumPy array")
@icontract.require(lambda volume, mask: volume.shape == mask.shape, "volume and mask shapes must match")
@icontract.require(lambda volume: volume.ndim == 3, "volume must be three-dimensional")
@icontract.require(lambda mask: bool(np.any(mask)), "mask must contain at least one selected voxel")
@icontract.require(lambda margin: margin >= 0, "margin must be non-negative")
@icontract.ensure(lambda volume, result: result[0].size <= volume.size, "crop must not be larger than the input")
@icontract.ensure(lambda result: len(result[1]) == 3, "bounds are reported for three axes")
def crop_to_mask(volume: np.ndarray, mask: np.ndarray, margin: int = 0) -> tuple[np.ndarray, SliceBounds3]:
    """Crop a 3D volume to the bounding box of non-zero mask voxels, including margin."""
    coordinates = np.argwhere(mask)
    lower = np.maximum(coordinates.min(axis=0) - int(margin), 0)
    upper = np.minimum(coordinates.max(axis=0) + int(margin) + 1, np.asarray(volume.shape))
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper, strict=True))
    bounds = tuple((int(lo), int(hi)) for lo, hi in zip(lower, upper, strict=True))
    return np.asarray(volume)[slices], bounds  # type: ignore[return-value]


@register_atom(witness_filter_small_components)
@icontract.require(lambda mask: isinstance(mask, np.ndarray), "mask must be a NumPy array")
@icontract.require(lambda mask: _is_binary_array(mask), "mask must be boolean or binary")
@icontract.require(lambda min_size: min_size > 0, "min_size must be positive")
@icontract.require(lambda structure: structure is None or isinstance(structure, np.ndarray), "structure must be None or a NumPy array")
@icontract.ensure(lambda mask, result: result.shape == mask.shape, "result preserves input shape")
@icontract.ensure(lambda result: _is_binary_array(result), "result must be binary")
def filter_small_components(mask: np.ndarray, min_size: int, structure: np.ndarray | None = None) -> np.ndarray:
    """Remove connected components smaller than a minimum voxel count from a binary mask."""
    binary = np.asarray(mask).astype(bool)
    labels, _ = ndimage.label(binary, structure=structure)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_size)
    keep[0] = False
    return keep[labels]


@register_atom(witness_parse_a3d_volume)
@icontract.require(lambda byte_stream: isinstance(byte_stream, bytes), "byte_stream must be bytes")
@icontract.require(lambda data_scale_factor: np.isfinite(data_scale_factor), "data_scale_factor must be finite")
@icontract.require(lambda shape_yzx: len(shape_yzx) == 3 and all(dim > 0 for dim in shape_yzx), "shape_yzx must contain three positive dimensions")
@icontract.require(
    lambda byte_stream, shape_yzx: len(byte_stream) == int(np.prod(shape_yzx)) * np.dtype(np.uint16).itemsize,
    "byte_stream length must match uint16 shape_yzx payload",
)
@icontract.ensure(lambda result: result.dtype == np.float32, "parsed volume is float32")
@icontract.ensure(lambda shape_yzx, result: result.shape == (shape_yzx[1], shape_yzx[0], shape_yzx[2]), "result is transposed to z-y-x")
def parse_a3d_volume(
    byte_stream: bytes,
    data_scale_factor: float,
    shape_yzx: tuple[int, int, int] = (660, 512, 512),
) -> np.ndarray:
    """Convert a memory-loaded Passenger Screening A3D uint16 payload into a z-y-x volume."""
    raw = np.frombuffer(byte_stream, dtype=np.uint16)
    volume_yzx = raw.reshape(shape_yzx)
    volume_zyx = np.transpose(volume_yzx, (1, 0, 2))
    return (volume_zyx.astype(np.float32) * np.float32(data_scale_factor)).astype(np.float32)

from __future__ import annotations

"""Ghost witnesses for medical-imaging preprocessing atoms."""

from sciona.ghost.abstract import AbstractArray, AbstractScalar


def witness_dicom_to_hounsfield(
    pixel_array: AbstractArray,
    slope: AbstractScalar,
    intercept: AbstractScalar,
) -> AbstractArray:
    """Model DICOM rescaling as shape-preserving intensity conversion."""
    return AbstractArray(shape=pixel_array.shape, dtype="float64")


def witness_dicom_window(
    pixel_array: AbstractArray,
    window_center: AbstractScalar,
    window_width: AbstractScalar,
    exact: AbstractScalar | None = None,
    y_min: AbstractScalar | None = None,
    y_max: AbstractScalar | None = None,
) -> AbstractArray:
    """Model VOI windowing as a shape-preserving bounded transform."""
    return AbstractArray(shape=pixel_array.shape, dtype="float64")


def witness_resample_volume(
    volume: AbstractArray,
    current_spacing: AbstractScalar,
    target_spacing: AbstractScalar,
    order: AbstractScalar | None = None,
) -> AbstractArray:
    """Model spacing resampling as a 3D array transform."""
    return AbstractArray(shape=volume.shape, dtype="float64")


def witness_macenko_stain_vectors(
    image: AbstractArray,
    alpha: AbstractScalar | None = None,
    beta: AbstractScalar | None = None,
    luminosity: AbstractScalar | None = None,
) -> AbstractArray:
    """Model Macenko estimation as a fixed 3-by-2 stain matrix."""
    return AbstractArray(shape=(3, 2), dtype="float64")


def witness_macenko_normalize(
    image: AbstractArray,
    target_stain_matrix: AbstractArray,
    alpha: AbstractScalar | None = None,
    beta: AbstractScalar | None = None,
    luminosity: AbstractScalar | None = None,
) -> AbstractArray:
    """Model stain normalization as preserving RGB image shape."""
    return AbstractArray(shape=image.shape, dtype="uint8")


def witness_max_intensity_projection(
    volume: AbstractArray,
    angle: AbstractScalar,
    rotation_axes: AbstractScalar | None = None,
    projection_axis: AbstractScalar | None = None,
    order: AbstractScalar | None = None,
) -> AbstractArray:
    """Model maximum-intensity projection as dropping one volume axis."""
    shape = volume.shape
    if len(shape) == 3:
        return AbstractArray(shape=(shape[1], shape[2]), dtype="float64")
    return AbstractArray(shape=(), dtype="float64")


def witness_extract_25d_slices(
    volume: AbstractArray,
    center_idx: AbstractScalar,
    num_adjacent: AbstractScalar,
    axis: AbstractScalar | None = None,
    mode: AbstractScalar | None = None,
) -> AbstractArray:
    """Model 2.5D extraction as a channel-first slice stack."""
    count = 1
    if num_adjacent.max_val is not None and num_adjacent.max_val >= 0:
        count = int(2 * num_adjacent.max_val + 1)
    tail = volume.shape[1:] if len(volume.shape) >= 1 else ()
    return AbstractArray(shape=(count, *tail), dtype=volume.dtype)


def witness_crop_to_mask(
    volume: AbstractArray,
    mask: AbstractArray,
    margin: AbstractScalar | None = None,
) -> AbstractArray:
    """Model mask cropping as returning a sub-volume."""
    return AbstractArray(shape=volume.shape, dtype=volume.dtype)


def witness_filter_small_components(
    mask: AbstractArray,
    min_size: AbstractScalar,
    structure: AbstractArray | None = None,
) -> AbstractArray:
    """Model component filtering as preserving mask shape."""
    return AbstractArray(shape=mask.shape, dtype="bool")


def witness_parse_a3d_volume(
    byte_stream: AbstractScalar,
    data_scale_factor: AbstractScalar,
    shape_yzx: AbstractScalar | None = None,
) -> AbstractArray:
    """Model A3D byte parsing as producing a z-y-x float volume."""
    return AbstractArray(shape=(512, 660, 512), dtype="float32")

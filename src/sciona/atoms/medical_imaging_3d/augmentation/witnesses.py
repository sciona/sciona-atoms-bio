from __future__ import annotations

"""Ghost witnesses for shape-preserving 3D augmentation atoms."""

from sciona.ghost.abstract import AbstractArray, AbstractScalar


def witness_random_rotate_3d(
    volume: AbstractArray,
    angles_deg: AbstractScalar,
    order: AbstractScalar | None = None,
) -> AbstractArray:
    """Model center-preserving rotation as preserving all array axes."""
    return AbstractArray(shape=volume.shape, dtype="float64")


def witness_elastic_deform_3d(
    volume: AbstractArray,
    sigma: AbstractScalar,
    alpha: AbstractScalar,
    seed: AbstractScalar,
    order: AbstractScalar | None = None,
) -> AbstractArray:
    """Model elastic deformation as preserving all array axes."""
    return AbstractArray(shape=volume.shape, dtype="float64")


def witness_scale_volume_3d(
    volume: AbstractArray,
    scale_factors: AbstractScalar,
    order: AbstractScalar | None = None,
) -> AbstractArray:
    """Model zoom plus center crop or padding as preserving all array axes."""
    return AbstractArray(shape=volume.shape, dtype="float64")


def witness_add_gaussian_noise_3d(
    volume: AbstractArray,
    mean: AbstractScalar,
    std: AbstractScalar,
    seed: AbstractScalar,
) -> AbstractArray:
    """Model additive seeded noise as preserving all array axes and dtype."""
    return AbstractArray(shape=volume.shape, dtype=volume.dtype)

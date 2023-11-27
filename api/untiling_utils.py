from typing import Tuple, Union, Optional

import numpy as np


def build_anomalous_map_from_patch_scores(error_scores: np.ndarray,
                                          patch_size: Union[Tuple[int, int], int],
                                          patch_coordinates: np.ndarray,
                                          step_size: Union[np.ndarray, int],
                                          forced_image_size: Optional[Tuple[int, int, int]] = None):
    """
    Given a list of values per patch, their positions (row, column), and a step_size per dimension
    reconstructs an image, averaging overlapping patches.

    :param error_scores: List of constant values that each patch will take over all its size.
    :param patch_size: Size of each patch.
    :param patch_coordinates: Coordinates of each of the patches (related to step_size, not to pixels). List of two ints.
    :param step_size: Step size between two consecutive patches. (height_step, width_step)
    :param forced_image_size: optional. Forces the output image to have this shape. (image_height, image_width, channels)
    :return: Rebuilt image.
    """
    if isinstance(patch_size, int):
        patch_size = [patch_size, patch_size]
    if len(patch_size) < 2:
        raise ValueError(f"Invalid value provided for patch_size: {patch_size}")

    error_patches = np.ones((len(error_scores), *patch_size[:2], 1), dtype=error_scores.dtype)
    error_patches *= error_scores.reshape((-1, 1, 1, 1))
    return rebuild_from_patches(error_patches,
                                patch_coordinates,
                                step_size,
                                forced_image_size=forced_image_size,
                                overlap=True)[..., 0]


def rebuild_from_patches(patches: np.ndarray,
                         patch_coordinates: np.ndarray,
                         step_size: Union[np.ndarray, int],
                         forced_image_size: Optional[Tuple[int, int, int]] = None,
                         overlap: bool = True):
    """
    Given a list of patches, their positions (row, column), and a step_size per dimension
    reconstructs the original image, with or without overlapping the patches with mean.

    :param patches: List of patches used to build the image. (num_patches, patch_height, patch_width, channels)
    :param patch_coordinates: Coordinates of each of the patches (related to step_size, not to pixels). List of two ints.
    :param step_size: Step size between two consecutive patches. (height_step, width_step)
    :param forced_image_size: optional. Forces the output image to have this shape. (image_height, image_width, channels)
    :param overlap: Flag to compute mean of overlapping regions.
    :return: Rebuilt image.
    """
    assert len(patches.shape) == 4, patches.shape
    assert len(patch_coordinates.shape) == 2 and patch_coordinates.shape[1] == 2, patch_coordinates.shape
    assert len(patch_coordinates) == len(patches)
    if isinstance(step_size, int):
        step_size = [step_size, step_size]
    step_size = np.asarray(step_size)
    assert len(step_size) >= 2, step_size
    step_size = step_size[:2]

    shape = forced_image_size
    if shape is None:
        max_per_dimension = np.max(patch_coordinates, axis=0)
        shape = (*(max_per_dimension * step_size + np.asarray(patches.shape[1:3])), patches.shape[-1])

    if overlap:
        return _rebuild_with_overlap(patches, patch_coordinates, step_size, shape)
    return _rebuild_without_overlap(patches, patch_coordinates, step_size, shape)


def _rebuild_without_overlap(patches, patch_coordinates, step_size, output_shape):
    result = np.zeros(output_shape, dtype=patches.dtype)
    for patch, coordinate in zip(patches, patch_coordinates):
        mask_slice = _get_slice_from_path_coordinate_and_step_size(coordinate, step_size, patches.shape[1:3])
        result[mask_slice] = patch
    return result


def _rebuild_with_overlap(patches, patch_coordinates, step_size, output_shape):
    result = np.zeros(output_shape, dtype=np.float32)
    overlapping_regions = np.zeros(output_shape, dtype=np.float32)
    for patch, coordinate in zip(patches, patch_coordinates):
        mask_slice = _get_slice_from_path_coordinate_and_step_size(coordinate, step_size, patches.shape[1:3])
        result[mask_slice] += patch
        overlapping_regions[mask_slice] += 1
    overlapping_regions = np.maximum(overlapping_regions, 1e-16)
    return (result / overlapping_regions).astype(patches.dtype)


def _get_slice_from_path_coordinate_and_step_size(coordinate, step_size, patch_size):
    return (
        slice(coordinate[0] * step_size[0], coordinate[0] * step_size[0] + patch_size[0]),
        slice(coordinate[1] * step_size[1], coordinate[1] * step_size[1] + patch_size[1])
    )
import logging
from typing import List, Optional, Tuple, Union

from tqdm import tqdm
import numpy as np


def prepare_keep_masks(masks: Optional[List[Optional[np.ndarray]]],
                       targets: List[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
    """Creates masks if masks is None or changes to bool type the existing masks."""
    if masks is None:
        return [None for _ in targets]
    else:
        return [x.astype(bool) if x is not None else None for x in masks]


def prepare_images_for_pixels_metric_computation(error_maps: List[np.ndarray],
                                                 targets: List[np.ndarray],
                                                 masks: Optional[List[Optional[np.ndarray]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the valid pixels of the targets and error maps using the ignore_masks.
    @param error_maps: List of test error maps. Each pixel is a distance measure.
    @param targets: Anomaly ground-truth.
    @param masks: List of masks that segment the valid pixels in the test error maps.
    @return: Tuple with (labels per pixel, predictions per pixel)
    """
    pixel_labels, pixel_predictions = [], []
    for index, (target, prediction) in tqdm(enumerate(zip(targets, error_maps)), desc='Preparing pixel level values.'):
        if masks is not None and masks[index] is not None:
            target = np.zeros_like(masks[index], dtype=np.uint8) if target is None else target
            mask = cut_pad_array_to_shape(masks[index], target.shape)
        else:
            target = np.zeros_like(prediction, dtype=np.uint8) if target is None else target
            mask = np.ones(target.shape, dtype=bool)
        prediction = cut_pad_array_to_shape(prediction, target.shape)

        pixel_labels.append(target[mask.astype(bool)])
        pixel_predictions.append(prediction[mask.astype(bool)])
    pixel_predictions = np.concatenate(pixel_predictions)
    pixel_labels = np.concatenate(pixel_labels)

    mask_finite_predictions = np.isfinite(pixel_predictions)
    percentage_finite = mask_finite_predictions.sum() / mask_finite_predictions.size
    pixel_predictions = pixel_predictions[mask_finite_predictions]
    pixel_labels = pixel_labels[mask_finite_predictions]
    if percentage_finite != 1:
        logging.warning(f'Only considering a {percentage_finite * 100: .1f}% of the pixels for the metric.')
    return (pixel_labels > 0).astype(np.float32), pixel_predictions


def cut_pad_array_to_shape(array: np.ndarray, dest_shape: tuple) -> np.ndarray:
    """
    Cuts or pads an array to a given shape.
    @param array: Array to process.
    @param dest_shape: Final shape.
    @return: Cut of padded array.
    """
    array = array[:dest_shape[0], :dest_shape[1]]
    return np.pad(array, [(0, dest_shape[0] - array.shape[0]), (0, dest_shape[1] - array.shape[1])])


def threshold_data(data: List[np.ndarray], threshold: float) -> List[np.ndarray]:
    """Thresholds a list of predictions to convert it to a binary 0 or 1 detection of anomalies."""
    # assume nan is 0
    result = []
    for image in data:
        tested = image.copy()
        tested[~np.isfinite(tested)] = 0
        result.append(tested >= threshold)
    return result


def normalize_error_maps_by_threshold(error_maps: List[np.ndarray],
                                      threshold: float,
                                      saturation_multiplier: float):
    return [normalize_error_map_by_threshold(x, threshold, saturation_multiplier) for x in error_maps]


def normalize_error_map_by_threshold(error_map: np.ndarray,
                                     threshold: float,
                                     saturation_multiplier: float):
    """
    Scales float image maps to a plottable representation.
    @param error_map: Image error map.
    @param threshold: Threshold above which a pixel is considered anomalous. It marks the value 127.5 in grayscale.
    @param saturation_multiplier: Adjusts the range of values above the threshold from grayscale 127.5 to 255. For
    instance, @saturation_multiplier = 1 implies that all values two times higher than the threshold are clipped to 255,
    @saturation_multiplier = 2 implies that all values three times higher than the threshold are clipped to 255, etc.
    @return: np.uint8 plottable map.
    """
    # Assume nan is 0
    error_map = error_map.copy()
    error_map[~np.isfinite(error_map)] = 0
    differences = error_map / threshold * 127.5
    over_threshold = differences > 127.5
    differences[over_threshold] = ((differences[over_threshold] - 127.5) / saturation_multiplier) + 127.5
    differences = np.clip(differences, 0, 255)
    differences = differences.astype(np.uint8)
    return differences


def get_binary_labels_from_targets(targets: List[Optional[np.ndarray]]) -> np.ndarray:
    """
    Gets the binary labels for an image given their two masks. An image is anomalous if it has an and_mask with some
    anomalous pixels. Warning: if an image has no and_mask but has a xor_mask that is non-zero, that means that the
    image labelling was unsuccessful and should be removed from the metrics. This images are marked with label -1.
    @param targets: Masks that segment the image region that both supervisors agreed to be anomalous.
    @return: Array of binary labels.
    """
    labels = []
    for target in targets:
        and_label = target is not None and np.sum(target) > 0
        labels.append(and_label)
    return np.asarray(labels, dtype=np.float32)

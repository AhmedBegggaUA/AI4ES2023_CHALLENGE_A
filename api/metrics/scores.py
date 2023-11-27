import logging
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from .metric_utils import cut_pad_array_to_shape, threshold_data


def get_auc(labels: np.ndarray,
            predictions: np.ndarray,
            batch_size: int = int(1e7),
            **kwargs):
    """
    Gets AUC ROC at pixel level by comparing each pixel to a mask. Only a large enough subset of the pixels is
    considered. Use @batch_size to configure the size.

    @param labels: np.array. 1D array with the targets.
    @param predictions: np.array. 1D array with the predictions.
    @param batch_size: int. Maximum number of labels to compute in a single auc operation (High numbers can explode in memory)
    @return: Evaluation value (0-1) expressing area under the roc curve
    """
    # Check there are at least two classes
    if np.max(labels) == np.min(labels):
        logging.error("[Get AUC] Cannot compute AUC with just one class")
        return np.nan
    results = get_subsample_auc_statistics(labels, predictions, batch_size, **kwargs)[:2]
    return metrics.auc(*results)


def get_subsample_auc_statistics(labels: np.ndarray,
                                 predictions: np.ndarray,
                                 batch_size: int = int(1e7),
                                 seed: Optional[int] = None,
                                 mode: str = 'roc') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes average curve auc of a random subsample.
    @param labels: np.arrays. Groundtruth for the auc.
    @param predictions: np.arrays. Predictions for the auc.
    @param batch_size: int. Maximum number of labels to compute in a single auc operation (High numbers can explode in memory)
    @param seed: Random seed
    @param mode: Type of curve to sample: roc or pr
    @return: Depending on the curve type returns: PR -> (recalls, precisions, thresholds). ROC -> (fprs, tprs, thresholds).
    """
    if seed is not None:
        np.random.seed(seed)
    order = np.arange(len(labels))
    np.random.shuffle(order)
    assert mode in ['roc', 'pr'], mode
    if mode == 'roc':
        return metrics.roc_curve(labels[order[:int(batch_size)]], predictions[order[:int(batch_size)]])
    precisions, recalls, thresholds = metrics.precision_recall_curve(labels[order[:int(batch_size)]],
                                                                     predictions[order[:int(batch_size)]])
    return recalls, precisions, thresholds


def get_best_point_in_precision_recall_curve(labels: np.ndarray,
                                             predictions: np.ndarray,
                                             batch_size: int = int(1e7),
                                             **kwargs) -> Tuple[float, float, float]:
    """
    Returns the TPr, FPr and threshold of the point in a ROC curve that maximizes a certain function. Currently, this
    function is the F1 score. The parameters are the same as the sklearn function roc_cuve.
    @return: TPr, FPr and threshold of the best value.
    """
    recall, precision, thresholds = get_subsample_auc_statistics(labels, predictions, batch_size,
                                                                 mode='pr', **kwargs)
    precision, recall = precision[:-1], recall[:-1]
    # precision, recall, thresholds = precision[1:-1], recall[1:-1], thresholds[1:]
    f1_scores = get_f1_score_from_precision_recall(precision, recall)
    best_position = np.argmax(f1_scores)
    return recall[best_position], precision[best_position], thresholds[best_position]


def get_precision_recall_from_threshold(precisions, recalls, thresholds, threshold):
    position = np.where(np.logical_and(threshold >= thresholds[:-1], threshold < thresholds[1:]))[0]
    return precisions[position], recalls[position]


def get_f1_score_from_precision_recall(precision: np.ndarray,
                                       recall: np.ndarray) -> np.ndarray:
    """Gets the f1 scores from a list of false positive rates and true positive rates."""
    assert isinstance(precision, np.ndarray)
    assert isinstance(recall, np.ndarray)
    return 2 * (precision * recall) / (precision + recall + 1e-7)


def false_positives_rate_at_true_positives_rate(labels: np.ndarray,
                                                predictions: np.ndarray,
                                                tpr_threshold: float):
    """
    Computes the false positive rate (fpr) at a true positive rate (tpr)
    Useful when the problem has already established working point restrictions

    @param labels: np.array. Numpy arrays have a shape (image_height, image_width). It is a binary image.
    @param predictions: np.array. Numpy arrays have equal shape and type as labels.
    @param tpr_threshold: float. True positive rate at which false positive rate will be calculated
    @return: false positive rate at stated true positive rate
    """
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
    return np.min(fpr[tpr >= tpr_threshold])


def get_image_predictions(error_maps: List[np.ndarray],
                          pixel_threshold: float,
                          image_threshold: float,
                          masks: Optional[List[Optional[np.ndarray]]] = None) -> np.ndarray:
    image_scores = get_binary_map_anomalous_score(error_maps, pixel_threshold, masks=masks)
    return (image_scores >= image_threshold).astype(np.float32)


def get_binary_map_anomalous_score(error_maps: List[np.ndarray],
                                   distance_threshold: float,
                                   masks: Optional[List[Optional[np.ndarray]]] = None) -> np.ndarray:
    """
    Gets the image-level anomaly scores for a list of anomalous maps and a distance threshold. The score is
    the amount of anomalous pixels per image.
    @param error_maps: List of error maps. Each element in the list is an image, so it must have the same shape as the
    targets. The value of each pixel is a distance measure, thus it should be the higher, the more anomalous.
    @param distance_threshold: Distance above which a pixel in the error map is considered anomalous.
    @param masks: List of keep masks to ignore some pixels in the error maps. It should be boolean masks. If pixel = True, consider that region.
    @return: Array with an anomalous score per image. Shape: (num_images, ). Type: float32.
    """
    binary_error_maps = threshold_data(error_maps, distance_threshold)
    num_anomalous_pixels = []
    for index, prediction in enumerate(binary_error_maps):
        if masks is not None:
            mask = masks[index] if masks[index] is not None else np.ones_like(prediction, dtype=bool)
            prediction = cut_pad_array_to_shape(prediction, mask.shape)
            prediction = prediction[mask.astype(bool)]
        num_anomalous_pixels.append(prediction.sum())
    return np.asarray(num_anomalous_pixels, dtype=np.float32)

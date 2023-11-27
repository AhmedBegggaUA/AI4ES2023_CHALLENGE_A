import logging
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from .metric_utils import prepare_keep_masks, prepare_images_for_pixels_metric_computation, \
    get_binary_labels_from_targets
from .scores import get_auc, get_best_point_in_precision_recall_curve, get_subsample_auc_statistics, \
    get_binary_map_anomalous_score, get_precision_recall_from_threshold, get_image_predictions


def test(error_maps: List[np.ndarray],
         targets: List[Optional[np.ndarray]],
         masks: Optional[List[Optional[np.ndarray]]] = None,
         threshold: Optional[float] = None,
         curve_type: str = 'pr',
         show_curves: bool = False) -> dict:
    """
    Gets the scores at pixel and image level for the predictions given.
    @param error_maps: List of error maps. Each element in the list is an image, so it must have the same shape as the
    targets. The value of each pixel is a distance measure, thus it should be the higher, the more anomalous.
    @param targets: Anomaly ground-truth. At pixel-level it takes any dtype; if > 0 it is anomalous.
    @param masks: List of keep masks to ignore some pixels in the error maps. It should be boolean masks. If pixel = True, consider that region.
    @param threshold: Allows to manually pass the threshold that will be used to calculate image AUC. Default: selects
    the best point in the PR curve at pixel level.
    @param curve_type: Pixel curve type. Default: pr. Alternative: roc.
    @param show_curves: Shows the AUC PR per pixel.
    @return: Dict with several useful info: pixel_auc, image_auc, threshold used for image_auc, pixel apriori and
    image apriori.
    """
    assert error_maps is not None
    assert targets is not None and any(target is not None for target in targets), targets
    assert curve_type in ['roc', 'pr']
    masks = prepare_keep_masks(masks, targets)
    pixel_metrics = get_pixel_metrics(error_maps=error_maps,
                                      targets=targets,
                                      masks=masks,
                                      show_curves=show_curves,
                                      curve_type=curve_type,
                                      fixed_threshold=threshold)

    pixel_metrics['pixel_threshold'] = threshold or pixel_metrics['pixel_threshold']
    image_metrics = get_global_metrics(error_maps=error_maps,
                                       distance_threshold=pixel_metrics['pixel_threshold'],
                                       targets=targets,
                                       masks=masks,
                                       show_curves=show_curves,
                                       curve_type=curve_type)
    return {**pixel_metrics, **image_metrics}


def get_pixel_metrics(error_maps: List[np.ndarray],
                      targets: List[np.ndarray],
                      masks: Optional[List[Optional[np.ndarray]]] = None,
                      show_curves: bool = False,
                      curve_type: str = 'roc',
                      fixed_threshold: Optional[float] = None) -> dict:
    """
    Gets the pixel AUC and some related metrics for the set of maps passed.
    @param error_maps: List of error maps. Each element in the list is an image, so it must have the same shape as the
    targets. The value of each pixel is a distance measure, thus it should be the higher, the more anomalous.
    @param targets: Anomaly ground-truth. At pixel-level it takes any dtype; if > 0 it is anomalous.
    @param masks: List of keep masks to ignore some pixels in the error maps. It should be boolean masks. If pixel = True, consider that region.
    @param curve_type: Pixel curve type. Default: pr. Alternative: roc.
    @param show_curves: Shows the AUC PR per pixel.
    @param fixed_threshold: Fixes a threshold instead of calculating the best point.
    @return: Dict with the info.
    """
    pixel_labels, pixel_predictions = prepare_images_for_pixels_metric_computation(error_maps, targets, masks)
    return get_context_info(pixel_labels,
                            pixel_predictions,
                            'pixel',
                            show=show_curves,
                            curve_type=curve_type,
                            fixed_threshold=fixed_threshold)


def get_global_metrics(error_maps: List[np.ndarray],
                       distance_threshold: float,
                       targets: List[np.ndarray],
                       masks: Optional[List[Optional[np.ndarray]]] = None,
                       show_curves: bool = False,
                       curve_type: str = 'roc') -> dict:
    """
    Gets the pixel AUC ROC and some related metrics from the set of maps passed.
    @param error_maps: List of error maps. Each element in the list is an image, so it must have the same shape as the
    targets. The value of each pixel is a distance measure, thus it should be the higher, the more anomalous.
    @param distance_threshold: Distance above which a pixel in the error map is considered anomalous.
    @param targets: Anomaly ground-truth. At pixel-level it takes any dtype; if > 0 it is anomalous.
    @param masks: List of keep masks to ignore some pixels in the error maps. It should be boolean masks. If pixel = True, consider that region.
    @param curve_type: Pixel curve type. Default: pr. Alternative: roc.
    @param show_curves: Shows the AUC PR per pixel.
    @return: Dict with the info.
    """
    image_labels = get_binary_labels_from_targets(targets)
    image_scores = get_binary_map_anomalous_score(error_maps, distance_threshold, masks=masks)
    image_results = get_context_info(image_labels, image_scores, 'image', show=show_curves, curve_type=curve_type)
    image_results["image_predictions"] = get_image_predictions(error_maps,
                                                               pixel_threshold=distance_threshold,
                                                               image_threshold=image_results["image_threshold"],
                                                               masks=masks)
    image_results["image_accuracy"] = np.mean(image_labels == image_results["image_predictions"])
    logging.info(f'Image accuracy: {image_results["image_accuracy"]}')
    return image_results


def get_context_info(labels: np.ndarray,
                     predictions: np.ndarray,
                     context_name: str,
                     show: bool = False,
                     curve_type: str = 'roc',
                     fixed_threshold: Optional[float] = None) -> dict:
    """
    Gets the pixel AUC ROC and some related metrics from the set of maps passed.
    @param labels: Anomaly ground-truth.
    @param predictions: Array with the predictions. Each pixel is a distance measure.
    @param context_name: Name of the level of the predictions and targets. Choices: Image or Pixel.
    @param show: Shows the curve PR curve for the current context level.
    @param curve_type: Pixel curve type. Default: pr. Alternative: roc.
    @param fixed_threshold: Fixes a threshold instead of calculating the best point.
    @return: Dict with the info.
    """
    percentage_w_defects = np.mean(labels) * 100
    amount_w_defects = int(np.sum(labels))
    auc = get_auc(labels, predictions, mode=curve_type)
    distance_threshold = fixed_threshold or get_best_point_in_precision_recall_curve(labels, predictions)[-1]

    if show:
        plot_curve(context_name, labels, predictions, distance_threshold, show=True)

    logging.info(f'{context_name.capitalize()}s with defects: {percentage_w_defects: .1f}%.')
    logging.info(f'{context_name.capitalize()}-wise AUC {curve_type.upper()}: {"%.3f" % auc}.')
    return {
        f'{context_name}_amount': len(labels),
        f'{context_name}_threshold': distance_threshold,
        f'{context_name}_auc': np.round(auc, 3),
        f'{context_name}_w_defects_%': np.round(percentage_w_defects, 1),
        f'{context_name}_w_defects': amount_w_defects,
    }


def plot_curve(context_name: str,
               labels: np.ndarray,
               predictions: np.ndarray,
               threshold: float,
               show: bool = True):
    recalls, precisions, thresholds = get_subsample_auc_statistics(labels, predictions, mode='pr')
    best_y, best_x = get_precision_recall_from_threshold(precisions, recalls, thresholds, threshold)
    ap = metrics.auc(recalls, precisions)
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(f'{context_name.capitalize()} AP: {ap:.3f}')
    fig.tight_layout()
    plt.plot(recalls, precisions)
    plt.scatter([best_x], [best_y], label='Threshold selected')
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    if show:
        plt.show()
        plt.close()
    return fig

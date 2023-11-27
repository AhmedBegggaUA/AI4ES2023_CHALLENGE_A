from typing import Tuple, Optional

import numpy as np
import cv2


def filter_background(data: np.ndarray,
                      image: np.ndarray,
                      dilate: int = 7,
                      start: Optional[Tuple[int, int, int]] = (240, 240, 240),
                      end: Optional[Tuple[int, int, int]] = (256, 256, 256),
                      fill_value: float = np.inf):
    """
    Sets the pixels that are background in the image to a @fill_value in the data provided. What it is considered as
    background can be configured with @start and @end parameters.
    @param data: Features extracted from the image.
    @param image: Image from which the features are extracted.
    @param dilate: Number of time to dilate the background to remove borders.
    @param start: Starting pixel color to be considered background.
    @param end: Ending pixel color to be considered background.
    @param fill_value: Value to set the background pixels in the @data.
    @return: Data with the background set to a certain value.
    """
    back = get_background(image, dilate, start, end)
    back = cv2.resize(back, data.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    datacopy = np.copy(data)
    datacopy[back != 0] = fill_value
    return datacopy


def get_background(image: np.ndarray,
                   dilate: int = 7,
                   start: Optional[Tuple[int, int, int]] = (240, 240, 240),
                   end: Optional[Tuple[int, int, int]] = (256, 256, 256)):
    assert start is not None or end is not None, f"{start} to {end}"
    if start is None:
        start = (0, 0, 0)
    if end is None:
        end = (256, 256, 256)

    back = cv2.inRange(image, tuple(start), tuple(end))
    if dilate > 0:
        back = cv2.dilate(back, np.ones((dilate, dilate), np.uint8), iterations=1)
    return cv2.resize(back, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

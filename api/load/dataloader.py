import glob
import logging
import os
from pathlib import Path
from typing import Optional, List, Union, Tuple

import numpy as np
from tqdm import tqdm
import cv2

from .dataset_utils import filter_background

_warned_paths = []


class DataLoader:
    def __init__(self, dataset_path: Union[str, Path]):
        """
        Class that returns the paths to image files of the Wood Kaggle Challenge dataset.
        @param dataset_path: Path to the wood dataset.
        """
        self.dataset_path = Path(dataset_path).expanduser().absolute()
        self.dataset_name = self.dataset_path.name.replace('-', '_')
        assert self.dataset_path.exists(), f'Dataset root directory passed does not exists: {str(self.dataset_path)}'

    def load(self,
             partition: Union[str, List[str]] = "",
             subset: int = 99999999) -> np.ndarray:
        """
        @param partition: Mode from which the paths should be loaded.
        @param subset: Number of paths to return.
        @return: List of image paths.
        """
        assert isinstance(subset, int)
        paths = []
        if isinstance(partition, str):
            paths = self._load_dataset(partition)
        elif isinstance(partition, List) or isinstance(partition, Tuple):
            for _partition in partition:
                paths += self._load_dataset(_partition)
        return np.asarray(paths)[:subset]

    def _load_dataset(self,
                      partition: str = "",
                      extension: str = '.png') -> List[str]:
        fullpath = self.dataset_path.joinpath(partition, 'images')
        if not fullpath.exists():
            return []

        current_names = sorted(os.listdir(str(fullpath)))
        names = [os.path.join(fullpath, x) for x in current_names]
        names = [path for path in names if extension in path]
        return names

    def find_images(self, image_names: List[str]) -> List[str]:
        """
        Finds images with the same name as the ones provided in the input list.

        @param image_names: List of image names to find.
        @return: List of the image paths found.
        """
        results = []
        for name in image_names:
            files = list(self.dataset_path.rglob(f"images/{name}*"))
            if len(files) == 0:
                logging.warning(f"The is no file with the provided name ({name}).")
                results.append(None)
                continue
            elif len(files) > 1:
                logging.warning(f"The are more than one files with the provided name ({name}): {files}")
            results.append(str(files[0]))
        return results

    @staticmethod
    def load_targets(image_paths: Union[np.ndarray, List],
                     preprocess: Optional = None,
                     verbose: bool = False,
                     directory_name: str = 'masks') -> List[np.ndarray]:
        """
        Returns the label information for a list of images in the Wood Kaggle Challenge dataset. Assumes preprocess
        returns a tensorflow tensor.
        @param image_paths: List of images from the Wood dataset.
        @param preprocess: Preprocess to apply to them.
        @type preprocess: Optional function.
        @param verbose: Blocks the tqdm if not verbose.
        @param directory_name:
        @return: Tuple of: list of target ground-truths, list of keep masks (the opposite of ignore XOR masks).
        """
        targets = []
        for path in tqdm(image_paths, desc='Loading masks', disable=not verbose):
            mask_path = DataLoader.get_ignore_mask_path_from_image(path, directory_name=directory_name)
            dirname = os.path.dirname(mask_path)
            if not os.path.exists(dirname) and dirname not in _warned_paths:
                logging.warning(f"The directory from where you are trying to load the masks does not exist: {dirname}")
                _warned_paths.append(dirname)
            target = DataLoader.load_target_mask_from_path(mask_path=mask_path,
                                                           image_path=path,
                                                           preprocess=preprocess)
            targets.append(target)
        return targets

    @staticmethod
    def load_keep_masks(image_paths: Union[np.ndarray, List],
                        verbose: bool = False,
                        directory_name: str = 'ignore_masks',
                        preprocess: Optional = None,
                        background_start: Optional[Tuple[int, int, int]] = None,
                        background_end: Optional[Tuple[int, int, int]] = None,
                        background_dilation: int = 0) -> List[np.ndarray]:
        """
        Returns the label information for a list of images in the Wood Kaggle Challenge dataset. Assumes preprocess
        returns a tensorflow tensor.
        @param image_paths: List of images from the Wood dataset.
        @param preprocess: Preprocess to apply to them.
        @type preprocess: Optional function.
        @param verbose: Blocks the tqdm if not verbose.
        @param directory_name: Name of the directory where ignore masks are stored.
        @param background_start: Colors BGR above which it is considered background. For instance: (246, 246, 246).
        @param background_end: Colors BGR below which it is considered background. For instance: (20, 20, 20).
        @param background_dilation: Dilation to apply to the background to remove adjacent pixels.
        @return: Tuple of: list of target ground-truths, list of keep masks (the opposite of ignore XOR masks).

        """
        masks = []
        for path in tqdm(image_paths, desc='Loading masks', disable=not verbose):
            mask_path = DataLoader.get_ignore_mask_path_from_image(path, directory_name=directory_name)
            keep_mask = DataLoader.load_keep_mask_from_path(mask_path=mask_path,
                                                            image_path=path,
                                                            preprocess=preprocess,
                                                            background_end=background_end,
                                                            background_dilation=background_dilation,
                                                            background_start=background_start)
            masks.append(keep_mask)
        return masks

    @staticmethod
    def load_target_mask_from_path(mask_path: str,
                                   image_path: Optional[str] = None,
                                   preprocess: Optional = None):
        if not os.path.exists(mask_path):
            image = cv2.imread(image_path)
            assert image is not None, "image_path not found"
            return np.zeros_like(image)[..., 0]

        target = DataLoader._load_mask(mask_path)
        if preprocess is not None:
            target = preprocess(target[..., np.newaxis])[..., 0].numpy()
        return target

    @staticmethod
    def load_keep_mask_from_path(mask_path: str,
                                 image_path: Optional[str] = None,
                                 preprocess: Optional = None,
                                 background_start: Optional[Tuple[int, int, int]] = None,
                                 background_end: Optional[Tuple[int, int, int]] = None,
                                 background_dilation: int = 0):
        if os.path.exists(mask_path):
            keep_mask = ~DataLoader._load_mask(mask_path)
            if preprocess is not None:
                keep_mask = preprocess(keep_mask[..., np.newaxis])[..., 0].numpy()
        else:
            assert image_path is not None
            image = cv2.imread(image_path)
            keep_mask = np.ones_like(image)[..., 0] * 255

        if background_start is not None or background_end is not None:
            assert image_path is not None
            image = cv2.imread(image_path)
            keep_mask = filter_background(image=image,
                                          data=keep_mask,
                                          dilate=background_dilation,
                                          start=background_start,
                                          end=background_end,
                                          fill_value=0)
        return keep_mask

    @staticmethod
    def get_target_path_from_image(image_path, directory_name: str = 'masks'):
        return DataLoader._right_replace(str(image_path), 'images', directory_name, 1)

    @staticmethod
    def get_ignore_mask_path_from_image(image_path, directory_name: str = 'ignore_masks'):
        return DataLoader._right_replace(str(image_path), 'images', directory_name, 1)

    @staticmethod
    def has_target_mask(image_path):
        return os.path.exists(DataLoader.get_target_path_from_image(image_path))

    @staticmethod
    def has_ignore_mask(image_path):
        return os.path.exists(DataLoader.get_ignore_mask_path_from_image(image_path))

    @staticmethod
    def _load_mask(path: str):
        """
        Loads a mask from disk.
        @param path: Path to the file that contains a mask.
        @return:
        """
        ''' reads the mask from the paths and binarizes it'''
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ret, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        return thresh

    @staticmethod
    def _right_replace(string, old, new, occurrences=0):
        """
        replaces old ocurrences in string by new from last character
        """
        li = string.rsplit(old, occurrences)
        return new.join(li)

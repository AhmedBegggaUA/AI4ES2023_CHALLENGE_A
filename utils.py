import tensorflow as tf
from tensorflow.keras.applications.resnet import  ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
from tqdm import tqdm
from skimage.util import view_as_windows


def get_feature_extractor(name_model):
    if name_model == "ResNet50":
        resnet = ResNet50(include_top=False, weights='imagenet')
        output = resnet.layers[80].output
        output = tf.keras.layers.GlobalMaxPool2D()(output)
        return tf.keras.Model(inputs=resnet.inputs, outputs=[output])
    elif name_model == "MobileNetV2":
        resnet = MobileNetV2(include_top=False, weights='imagenet')
        output = resnet.layers[80].output
        output = tf.keras.layers.GlobalMaxPool2D()(output)
        return tf.keras.Model(inputs=resnet.inputs, outputs=[output])
    else:
        print("Model not found! Please choose a valid model.")
        exit()

''' 
This function extracts windows from the images and returns them as a numpy array.
'''
def extract_windows(paths, window_size, step_size):
    """Extract fixed size windows from variable size images."""
    images = [cv2.imread(path)[..., ::-1] for path in tqdm(paths, desc="Loading images", leave=False)]
    return [view_as_windows(image, window_size, step=step_size) for image in images]    
    
def extract_features(paths,window_size,step_size,batch_size,feature_extractors, *args, **kwargs):    
    """
    :return: A tuple with:
     * Window features.
     * Windows.
     * Position of each window in the image.
     * Per window identifier of the index of their corresponding image.
    """
    windows = extract_windows(paths,window_size,step_size, *args, **kwargs)
    image_ids = np.concatenate([[index for row in range(sample.shape[0]) for col in range(sample.shape[1])] for index, sample in enumerate(windows)])
    positions = np.concatenate([[[row, col] for row in range(sample.shape[0]) for col in range(sample.shape[1])] for sample in windows])
    windows = np.concatenate([sample.reshape(-1, *window_size) for sample in windows], axis=0)

    # Filter completely white patches
    mask = windows.min(axis=(1, 2, 3)) < 240 # Filter completely white patches, pixels with value lower than 240 are considered white
    windows = windows[mask] # Filter windows, the ones that are not white
    windows = windows.astype(np.float32)
    windows = windows / 255.0
    image_ids = image_ids[mask] # Filter image ids, the ones that are not white
    positions = positions[mask] # Filter positions, the ones that are not white
    batch_size = 2048*2*2
    features = []
    if len(feature_extractors) == 1:
        feature_extractor = get_feature_extractor(feature_extractors[0])
        for i in tqdm(range(0, len(windows), batch_size), desc="Extracting features", leave=False):
            batch = preprocess_input(windows[i:i + batch_size].copy())
            batch_features = feature_extractor.predict(batch, verbose=False)
            features.append(batch_features.copy())
    else:
        feature_extractor = get_feature_extractor(feature_extractors[0])
        feature_extractor2 = get_feature_extractor(feature_extractors[1])
        for i in tqdm(range(0, len(windows), batch_size), desc="Extracting features", leave=False):
            batch = preprocess_input(windows[i:i + batch_size].copy())
            batch_features_movil = feature_extractor.predict(batch, verbose=False)
            batch_features_resnet = feature_extractor2.predict(batch, verbose=False)
            batch_features = np.concatenate([batch_features_movil.copy(),batch_features_resnet.copy()],axis=1)
            features.append(batch_features.copy())
    features = np.concatenate(features, axis=0)
    return features, windows, positions, image_ids
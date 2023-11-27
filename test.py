import cv2
import numpy as np
from tqdm import tqdm
from api.load import DataLoader
from api.metrics import test as test_metrics
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet import  ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from joblib import dump, load
from api.untiling_utils import build_anomalous_map_from_patch_scores
from utils import *
def test(model_EMU = ["MobileNetV2","ResNet50"],model_ANOMALY = "comb",window_size=(16,16,3), step_size=8, batch_size=2048):
    dataloader = DataLoader(dataset_path="dataset/") # Load data from dataset folder
    test_paths = dataloader.load("test") # Load train paths
    print("Total amount of testing samples: ", len(test_paths)) # Print the amount of training samples    
    try:
        if model_EMU == "None":
            # Lanzamos una excepción para que no se ejecute el código de abajo
            raise Exception("No features found")
        test_features = np.load("test_features.npy")
        test_coordinates = np.load("test_coordinates.npy")
        test_image_ids = np.load("test_image_ids.npy")
        print("Features loaded from disk successfully!")
    except:
        print("No features found, extracting features from test dataset...")
        test_features, _, test_coordinates, test_image_ids = extract_features(paths = test_paths, window_size=window_size, step_size=step_size, batch_size=batch_size, feature_extractors=model_EMU)
        if model_EMU != None:
            np.save("test_features.npy", test_features)
            np.save("test_coordinates.npy", test_coordinates)
            np.save("test_image_ids.npy", test_image_ids)
        if test_features == None:
            test_features = np.zeros((6148907,512))
    print("Shape of the test features dataset: ", test_features.shape)
    print("Loading the model... " + model_ANOMALY)
    if model_ANOMALY == "IF":
        model = load("IF.joblib")
    elif model_ANOMALY == "AE":
        model = load("AE.h5")
    elif model_ANOMALY == "comb":
        model_IF = load("IF.joblib")
        model_AE = load("AE.h5")
    else:
        print("Model not found! Please choose a valid model.")
        print("Bye!")
        exit()
    batch_size = batch_size
    if model_ANOMALY != "comb":
        test_scores = []
        for i in tqdm(range(0, len(test_features), batch_size), desc="Extracting features", leave=False):
            batch = preprocess_input(test_features[i:i + batch_size].copy())
            batch_score = model.predict_proba(batch)[:,1]
            test_scores.append(batch_score.copy())
        test_scores = np.concatenate(test_scores, axis=0)
        print("Saving scores...")
        np.save("test_scores_"+model_ANOMALY+".npy", test_scores)
    else:
        test_scores_IF = []
        test_scores_AE = []
        for i in tqdm(range(0, len(test_features), batch_size), desc="Extracting features", leave=False):
            batch = preprocess_input(test_features[i:i + batch_size].copy())
            batch_score_IF = model_IF.predict_proba(batch)[:,1]
            batch_score_AE = model_AE.predict(batch)[:,1]
            test_scores_IF.append(batch_score_IF.copy())
            test_scores_AE.append(batch_score_AE.copy())
        test_scores_IF = np.concatenate(test_scores_IF, axis=0)
        test_scores_AE = np.concatenate(test_scores_AE, axis=0)
        test_scores = np.max([test_scores_IF,test_scores_AE],axis=0)
        print("Saving scores...")
        np.save("test_scores_IF.npy", test_scores_IF)
        np.save("test_scores_AE.npy", test_scores_AE)
    print("Predicting labels...")
    # free some memory
    del test_features
    del model
    predictions = []
    test_shapes = [cv2.imread(path).shape for path in test_paths]
    for image_id in np.unique(test_image_ids):
        mask = test_image_ids == image_id
        
        predictions.append(build_anomalous_map_from_patch_scores(test_scores[mask], 
                                                                window_size, 
                                                                patch_coordinates=test_coordinates[mask], 
                                                                step_size=step_size, 
                                                                forced_image_size=test_shapes[image_id]))
    test_targets = dataloader.load_targets(test_paths)
    test_masks = dataloader.load_keep_masks(test_paths, background_start=(240, 240, 240), background_dilation=31)
    results = test_metrics(error_maps=predictions, targets=test_targets, masks=test_masks)
    print("Results for model: ", model_ANOMALY, " and EMU: ", model_EMU)
    print("Pixel AP score: ", results["pixel_auc"])
    print("Image AP score: ", results["image_auc"])
    print(f"Pixel a-priori probability: {results['pixel_w_defects_%']}%")
    print(f"Image a-priori probability: {results['image_w_defects_%']}%")
    return True
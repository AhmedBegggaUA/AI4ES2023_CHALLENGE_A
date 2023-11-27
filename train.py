import cv2
import numpy as np
from tqdm import tqdm
from api.load import DataLoader
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet import  ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from joblib import dump, load
from utils import *
def train(model_EMU = ["MobileNetV2","ResNet50"],model_ANOMALY = "comb",window_size=16, step_size=8, batch_size=2048):
    dataloader = DataLoader(dataset_path="dataset/") # Load data from dataset folder
    train_paths = dataloader.load("train") # Load train paths
    print("Total amount of training samples: ", len(train_paths)) # Print the amount of training samples    
    try:
        train_features = np.load("train_features.npy")
        print("Features loaded from disk successfully!")
    except:
        print("No features found, extracting features from training dataset...")
        train_features, _, _, _ = extract_features(paths = train_paths, window_size=window_size, step_size=step_size, batch_size=batch_size, feature_extractors=model_EMU)
        np.save("train_features.npy", train_features)

    print("Shape of the training features dataset: ", train_features.shape)
    print("Training the model... " + model_ANOMALY)
    if model_ANOMALY == "IF":
        model = IForest(n_estimators=100, contamination=0.1, n_jobs=-1)
    elif model_ANOMALY == "AE":
        model = AutoEncoder(epochs=50,hidden_neurons=[128, 64,32,32,64,128],batch_size=1024,verbose=1,validation_size=0.1)
    elif model_ANOMALY == "comb":
        model_IF = IForest(n_estimators=100, contamination=0.1, n_jobs=-1)
        model_AE = AutoEncoder(epochs=50,hidden_neurons=[128, 64,32,32,64,128],batch_size=1024,verbose=1,validation_size=0.1)
    else:
        print("Model not found! Please choose a valid model.")
        print("Bye!")
        exit()
    batch_size = batch_size
    if model_ANOMALY != "comb":
        for i in tqdm(range(0, len(train_features), batch_size), desc="Extracting features", leave=False):
            batch = preprocess_input(train_features[i:i + batch_size].copy())
            model.fit(batch)
    else:
        for i in tqdm(range(0, len(train_features), batch_size), desc="Extracting features", leave=False):
            batch = preprocess_input(train_features[i:i + batch_size].copy())
            model_IF.fit(batch)
            model_AE.fit(batch)
    if model_ANOMALY == "IF":
        dump(model, "IF.joblib")
    elif model_ANOMALY == "AE":
        model.save("AE.h5")
    else:
        dump(model_IF, "IF.joblib")
        dump(model_AE, "AE.h5")
    print("Model saved, Bye!")
    return True
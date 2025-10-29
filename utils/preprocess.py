"""
preprocess.py
-------------
Handles input preprocessing before making predictions.
"""

import numpy as np
import pickle
import os

MODEL_DIR = os.path.join(os.getcwd(), 'model')


def load_scaler():
    """Load saved StandardScaler."""
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler file not found at 'model/scaler.pkl'")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler


def validate_input(data):
    """
    Validate incoming data.
    Args:
        data (list or dict): Feature values.
    Returns:
        np.ndarray: Numpy array of numeric values.
    """
    if isinstance(data, dict):
        values = list(data.values())
    elif isinstance(data, list):
        values = data
    else:
        raise ValueError("Input must be a list or dictionary.")

    try:
        values = [float(v) for v in values]
    except ValueError:
        raise ValueError("All input values must be numeric.")

    return np.array([values])  # shape (1, n_features)


def preprocess_input(data):
    """
    Preprocess raw input before prediction.
    - Validates data
    - Applies scaling
    Returns scaled feature array.
    """
    scaler = load_scaler()
    features = validate_input(data)
    scaled_features = scaler.transform(features)
    return scaled_features

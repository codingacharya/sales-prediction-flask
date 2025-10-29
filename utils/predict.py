"""
predict.py
-----------
Loads the trained model and makes predictions from preprocessed input.
"""

import pickle
import os
from utils.preprocess import preprocess_input

MODEL_DIR = os.path.join(os.getcwd(), 'model')


def load_model():
    """Load saved ML model."""
    model_path = os.path.join(MODEL_DIR, 'model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found at 'model/model.pkl'")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict_sales(input_data):
    """
    Make prediction using the trained model.
    Args:
        input_data (list or dict): Raw features (TV, Radio, Newspaper)
    Returns:
        float: Predicted sales value.
    """
    model = load_model()
    scaled_features = preprocess_input(input_data)
    prediction = model.predict(scaled_features)
    return round(float(prediction[0]), 2)

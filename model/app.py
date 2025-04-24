import json
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import load_model
import logging
import os

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define available challenges
available_challenges = [
    {
        "ChallengeName": "Weight Loss Combo",
        "type": "cardio",
        "desc": "High-intensity interval training for fat burning",
        "Exercises": ["jumping jacks", "burpees", "mountain climbers"],
        "Intensity": "most"
    },
    {
        "ChallengeName": "Fitness Starter",
        "type": "combo",
        "desc": "Beginner-friendly full-body workout routine",
        "Exercises": ["bodyweight squats", "modified pushups", "plank holds"],
        "Intensity": "less"
    }
]
# Initialize scalers and separate encoders for each categorical feature
scaler = StandardScaler()
challenge_name_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
intensity_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Load the model
try:
    model = load_model('cbf_exercise_model.h5')
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    # For demonstration/testing, create a dummy model that returns random scores
    class DummyModel:
        def predict(self, X, verbose=0):
            return np.random.random(size=(X.shape[0], 1))
    model = DummyModel()
    logging.warning("Using dummy model for testing")

# Initialize preprocessors
def initialize_preprocessors():
    # Sample data with numerical features
    numerical_data = np.array([
        [170, 70, 30, 1],  # Male
        [165, 60, 25, 0],  # Female
        [180, 80, 40, 1],  # Male
        [155, 55, 35, 0]   # Female
    ])
    
    # Fit scaler on numerical features
    scaler.fit(numerical_data)
    
    # Extract categorical data
    challenge_names = np.array([challenge["ChallengeName"] for challenge in available_challenges]).reshape(-1, 1)
    challenge_types = np.array([challenge["type"] for challenge in available_challenges]).reshape(-1, 1)
    intensity_levels = np.array([challenge["Intensity"] for challenge in available_challenges]).reshape(-1, 1)
    
    # Fit separate encoders for each categorical feature
    challenge_name_encoder.fit(challenge_names)
    type_encoder.fit(challenge_types)
    intensity_encoder.fit(intensity_levels)

# Initialize preprocessors
initialize_preprocessors()

# Function to preprocess user data and generate recommendations
def predict_recommendations(user_data):
    # Prepare user features
    user_features = np.array([
        user_data['height'],
        user_data['weight'],
        user_data['age'],
        1 if user_data['gender'].lower() == 'male' else 0
    ]).reshape(1, -1)
    
    # Scale user features
    user_features_scaled = scaler.transform(user_features)
    
    # Generate predictions for each available challenge
    predictions = []
    
    for challenge in available_challenges:
        # Encode challenge features
        challenge_encoded = challenge_name_encoder.transform([[challenge['ChallengeName']]])
        type_encoded = type_encoder.transform([[challenge['type']]])
        intensity_encoded = intensity_encoder.transform([[challenge['Intensity']]])
        
        # Combine features
        features = np.hstack([
            user_features_scaled,
            challenge_encoded,
            type_encoded,
            intensity_encoded
        ])
        
        # Predict score
        score = model.predict(features, verbose=0)[0][0]
        
        # Add to predictions
        challenge_copy = challenge.copy()
        challenge_copy['RecommendationScore'] = float(score)
        predictions.append(challenge_copy)
    
    # Sort by predicted score
    predictions.sort(key=lambda x: x['RecommendationScore'], reverse=True)
    
    return predictions

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get user data from request
        user_data = request.json
        
        # Validate required fields
        required_fields = ['height', 'weight', 'age', 'gender']
        for field in required_fields:
            if field not in user_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate data types
        if not isinstance(user_data['height'], (int, float)) or not isinstance(user_data['weight'], (int, float)):
            return jsonify({'error': 'Height and weight must be numeric.'}), 400
        if not isinstance(user_data['age'], int):
            return jsonify({'error': 'Age must be an integer.'}), 400
        if user_data['gender'].lower() not in ['male', 'female']:
            return jsonify({'error': 'Gender must be "male" or "female".'}), 400
        
        # Generate recommendations
        recommendations = predict_recommendations(user_data)
        
        # Return recommendations as JSON
        return jsonify({
            'user_data': user_data,
            'recommendations': recommendations
        })
    
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

if __name__ == '__main__':
    app.run(debug=True)



    
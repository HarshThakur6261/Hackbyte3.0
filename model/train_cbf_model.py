import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Step 2: Load the JSON Dataset
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Step 3-4: Preprocess and Vectorize Data
def preprocess_data(data):
    # Extract input features and target values
    input_features = []
    target_scores = []
    
    for entry in data:
        input_data = entry['input_data']
        
        # For each recommendation in the output, create a training example
        for recommendation in entry['output_data']['recommendations']:
            # Input features: user characteristics
            features = [
                input_data['height'],
                input_data['weight'],
                input_data['age'],
                1 if input_data['gender'] == 'male' else 0,  # Simple encoding for gender
                recommendation['ChallengeName'],
                recommendation['type'],
                recommendation['Intensity']
            ]
            
            input_features.append(features)
            target_scores.append(recommendation['RecommendationScore'])
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(input_features, columns=[
        'height', 'weight', 'age', 'gender', 'challenge_name', 'type', 'intensity'
    ])
    
    # Separate numerical and categorical features
    numerical_features = df[['height', 'weight', 'age', 'gender']].values
    
    # One-hot encode categorical features
    import sklearn
    from packaging import version

    if version.parse(sklearn.__version__) >= version.parse('1.2.0'):
        encoder = OneHotEncoder(sparse_output=False)
    else:
        encoder = OneHotEncoder(sparse=False)
    challenge_encoded = encoder.fit_transform(df[['challenge_name']])
    type_encoded = encoder.fit_transform(df[['type']])
    intensity_encoded = encoder.fit_transform(df[['intensity']])
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features_scaled = scaler.fit_transform(numerical_features)
    
    # Combine all features
    X = np.hstack([
        numerical_features_scaled, 
        challenge_encoded, 
        type_encoded, 
        intensity_encoded
    ])
    
    y = np.array(target_scores)
    
    return X, y, scaler, encoder

# Step 5: Build Neural Network Model
def build_model(input_shape):
    inputs = Input(shape=(input_shape,))
    
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

# Step 6-7: Split Data and Train Model
def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Build model
    model = build_model(X_train.shape[1])
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=16,
        verbose=1
    )
    
    # Evaluate model
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return model, X_test, y_test

# Step 8: Predict for New Items
def predict_recommendations(model, scaler, encoder, user_data, available_challenges):
    # Prepare user features
    user_features = np.array([
        user_data['height'],
        user_data['weight'],
        user_data['age'],
        1 if user_data['gender'] == 'male' else 0
    ]).reshape(1, -1)
    
    # Scale user features
    user_features_scaled = scaler.transform(user_features)
    
    # Generate predictions for each available challenge
    predictions = []
    
    for challenge in available_challenges:
        # Encode challenge features
        challenge_encoded = encoder.transform([[challenge['ChallengeName']]])
        type_encoded = encoder.transform([[challenge['type']]])
        intensity_encoded = encoder.transform([[challenge['Intensity']]])
        
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
        challenge_copy['PredictedScore'] = float(score)
        predictions.append(challenge_copy)
    
    # Sort by predicted score
    predictions.sort(key=lambda x: x['PredictedScore'], reverse=True)
    
    return predictions

def main():
    # Load data
    data = load_data('exercise_dataset_200_improved.json')
    
    # Preprocess data
    X, y, scaler, encoder = preprocess_data(data)
    
    # Train model
    model, X_test, y_test = train_model(X, y)
    
    # Save model
    model.save('cbf_exercise_model.h5')
    
    # Example prediction
    new_user = {
        'height': 175,
        'weight': 70,
        'age': 30,
        'gender': 'male',
        'goal': 'fitness'
    }
    
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
    
    recommendations = predict_recommendations(model, scaler, encoder, new_user, available_challenges)
    
    print("\nRecommendations for new user:")
    for i, rec in enumerate(recommendations):
        print(f"{i+1}. {rec['ChallengeName']} (Score: {rec['PredictedScore']:.2f})")
        print(f"   Type: {rec['type']}, Intensity: {rec['Intensity']}")
        print(f"   Description: {rec['desc']}")
        print(f"   Exercises: {', '.join(rec['Exercises'])}")
        print()

if __name__ == "__main__":
    main()

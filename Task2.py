
# ========================================
# TASK 2: Emotion Recognition from Speech
# ========================================

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import os

def extract_features(audio_path):
    """Extract MFCC features from audio file"""
    try:
        # Load audio file
        audio, sr = librosa.load(audio_path, duration=3, offset=0.5)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        return mfccs_scaled
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def emotion_recognition_model():
    """Simple Emotion Recognition Model"""
    
    print("Emotion Recognition from Speech")
    print("=" * 50)
    
    # Since we don't have actual audio files, let's create sample data
    # In real implementation, you would load actual audio files
    np.random.seed(42)
    n_samples = 500
    
    # Simulate MFCC features (13 features)
    X = np.random.randn(n_samples, 13)
    
    # Simulate emotions (0: happy, 1: sad, 2: angry, 3: neutral)
    emotions = ['happy', 'sad', 'angry', 'neutral']
    y = np.random.randint(0, 4, n_samples)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Emotions distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Results
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=emotions))
    
    return model, scaler, accuracy
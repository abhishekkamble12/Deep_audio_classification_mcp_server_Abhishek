import joblib

# Save the trained LSTM model
model.save("audio_deepfake_model.h5")

# Save the StandardScaler
joblib.dump(scaler, "audio_scaler.pkl")

import os
import librosa
import numpy as np
import tensorflow as tf
import joblib
from mcp.server.fastmcp import FastMCP

# Initialize the FastMCP server
mcp = FastMCP("AudioDeepfakeDetector")

# Global variables for model and scaler
MODEL_PATH = "audio_deepfake_model.h5"
SCALER_PATH = "audio_scaler.pkl"
model = None
scaler = None

def load_resources():
    """Load the trained model and scaler into memory."""
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and Scaler loaded successfully.")
    else:
        print(f"Warning: Ensure {MODEL_PATH} and {SCALER_PATH} are in the current directory.")

def extract_features(file_path):
    """Extract MFCC features exactly as done in the training notebook."""
    # Load audio (resampled to 16kHz as per deepfake detection standard)
    audio, sr = librosa.load(file_path, sr=16000)
    
    # Extract MFCCs (40 coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    # Take the mean across time to get a fixed-size 1D vector
    return np.mean(mfccs.T, axis=0)

@mcp.tool()
def analyze_audio(file_path: str) -> str:
    """
    Analyzes an audio file (.wav, .mp3) to determine if it is Real or a Deepfake.
    
    Args:
        file_path: Absolute path to the audio file on the local file system.
        
    Returns:
        A formatted string describing the prediction result (Real or Fake) and the confidence score.
    """
    if model is None or scaler is None:
        return "Error: Model or scaler not loaded. Please ensure the h5 and pkl files are present."
        
    if not os.path.exists(file_path):
        return f"Error: Could not find audio file at {file_path}"
        
    try:
        # 1. Feature Extraction
        features = extract_features(file_path)
        
        # 2. Scaling (using the pre-trained StandardScaler)
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # 3. Reshape for LSTM (samples=1, timesteps=1, features=40)
        features_reshaped = features_scaled.reshape(1, 1, 40)
        
        # 4. Prediction
        prediction = model.predict(features_reshaped, verbose=0)[0][0]
        
        # 5. Format Results (Label 1 = Fake, Label 0 = Real)
        result = "Fake" if prediction > 0.5 else "Real"
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        
        return f"Analysis Complete!\nResult: {result}\nConfidence: {confidence:.2%}\nPath: {file_path}"
        
    except Exception as e:
        return f"An error occurred during audio analysis: {str(e)}"

if __name__ == "__main__":
    # Load the machine learning assets before starting the server
    load_resources()
    
    # Run the MCP server over standard input/output
    mcp.run()
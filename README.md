# Deepfake Audio Detection MCP Server

This project bridges a locally trained Deepfake Audio Detection model (LSTM-based) with AI assistants using the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). By running this server, you can give AI assistants (like Claude) the ability to natively analyze audio files on your local machine and determine if they are real or AI-generated deepfakes.

## Features
* **MFCC Feature Extraction**: Uses `librosa` to extract 40 Mel-frequency cepstral coefficients (MFCCs) from input audio files.
* **LSTM Inference pipeline**: Uses a pre-trained TensorFlow/Keras LSTM model to classify the audio.
* **FastMCP Integration**: Exposes the Python inference pipeline as a standard MCP Tool (`analyze_audio`) over standard input/output.


## Prerequisites

Ensure you have Python 3.8+ installed on your system. You will need the following Python libraries:
* `mcp` (FastMCP SDK)
* `tensorflow`
* `librosa`
* `scikit-learn`
* `numpy`
* `joblib`

## Installation & Setup

### 1. Export the Model and Scaler
Before running the server, you need to export your trained LSTM model and `StandardScaler` from your Jupyter Notebook (`Audio.ipynb`). 

Run the following code in a new cell at the end of your notebook:

```python
import joblib

# Save the trained LSTM model
model.save("audio_deepfake_model.h5")

# Save the StandardScaler
joblib.dump(scaler, "audio_scaler.pkl")

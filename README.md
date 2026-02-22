🎙️ Deepfake Audio Detection MCP Server

This project bridges a locally trained Deepfake Audio Detection (LSTM-based) model with AI assistants using the Model Context Protocol (MCP)
.

By running this server, AI assistants (such as Claude Desktop) can securely analyze audio files on your local machine and determine whether they are:

✅ Real (Human Voice)

⚠️ AI-Generated (Deepfake)

🚀 Features

🎵 MFCC Feature Extraction
Uses librosa to extract 40 Mel-Frequency Cepstral Coefficients (MFCCs) from input audio.

🧠 LSTM Inference Pipeline
A trained TensorFlow/Keras LSTM model classifies the audio.

🔌 FastMCP Integration
Exposes the inference pipeline as an MCP Tool (analyze_audio) via standard input/output.

📊 Streamlit Web App (Optional UI)
User-friendly interface for uploading and analyzing audio files.

🧰 Tech Stack

Python 3.8+

TensorFlow / Keras

Librosa

NumPy

Scikit-learn

Joblib

MCP (FastMCP SDK)

Streamlit (for UI)

📦 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/deepfake-audio-mcp.git
cd deepfake-audio-mcp
2️⃣ Install Dependencies
pip install mcp tensorflow librosa scikit-learn numpy joblib streamlit
3️⃣ Export the Model and Scaler

Before running the MCP server, export your trained LSTM model and StandardScaler from your Jupyter Notebook (Audio.ipynb).

Add this cell at the end of your notebook:

import joblib

# Save the trained LSTM model
model.save("audio_deepfake_model.h5")

# Save the StandardScaler
joblib.dump(scaler, "audio_scaler.pkl")

After running this, ensure the following files exist in your project directory:

audio_deepfake_model.h5

audio_scaler.pkl

▶️ Running the MCP Server

Start the MCP server:

python server.py

This exposes the tool:

analyze_audio(file_path: str) -> { prediction, confidence }

AI assistants connected via MCP can now analyze local audio files.

🌐 Running the Streamlit App (Optional)

If you built the UI:

streamlit run app.py

Then open:

http://localhost:8501
🖼️ Screenshots
Streamlit App – Upload Interface
<img width="1907" height="866" alt="Upload Interface" src="https://github.com/user-attachments/assets/8fc4226d-61b0-4189-ac81-425e7506e1a2" />
Streamlit App – Prediction Result
<img width="1389" height="855" alt="Prediction Result" src="https://github.com/user-attachments/assets/c267811c-0b67-4ac4-bd66-fec67ba4bcf6" />
🧠 How It Works

Audio file is loaded using librosa

40 MFCC features are extracted

Features are normalized using StandardScaler

Data is reshaped for LSTM input

Model predicts:

0 → Real

1 → Deepfake

Confidence score is returned

📌 Example Output
{
  "prediction": "Deepfake",
  "confidence": 0.93
}
🔒 Why MCP?

Using the Model Context Protocol (MCP) allows:

Secure local file access

Standardized AI tool integration

No cloud upload required

Seamless AI assistant interaction

📜 License

This project is open-source and available under the MIT License.

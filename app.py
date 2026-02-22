import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_PATH = SCRIPT_DIR / 'deepfake_detection_model.h5'
SCALER_PATH = SCRIPT_DIR / 'scaler.pkl'


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(str(MODEL_PATH))
    scaler = joblib.load(str(SCALER_PATH))
    return model, scaler


def extract_features(file):
    audio, sr = librosa.load(file, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0), audio, sr


def main():
    st.set_page_config(page_title="Audio Deepfake Detector", page_icon="🎙️", layout="centered")
    st.title("🎙️ Audio Deepfake Detector")
    st.markdown(
        "Upload a **.wav** or **.mp3** file and the model will predict whether "
        "the audio is **Real** or **AI-generated (Fake)**."
    )

    model, scaler = load_assets()

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_file is None:
        st.info("👆 Upload an audio file to get started.")
        return

    st.audio(uploaded_file, format="audio/wav")

    if not st.button("🔍 Analyze Audio", use_container_width=True):
        return

    with st.spinner("Extracting features and predicting…"):
        try:
            features, audio, sr = extract_features(uploaded_file)
        except Exception as e:
            st.error(f"Failed to process audio: {e}")
            return

        features_scaled = scaler.transform(features.reshape(1, -1))
        features_reshaped = features_scaled.reshape(1, 1, 40)

        prediction = model.predict(features_reshaped, verbose=0)[0][0]

    is_fake = prediction > 0.5
    label = "Fake (AI-Generated)" if is_fake else "Real (Authentic)"
    confidence = float(prediction if is_fake else 1 - prediction)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if is_fake:
            st.error(f"🔴 **{label}**")
        else:
            st.success(f"🟢 **{label}**")

    with col2:
        st.metric("Confidence", f"{confidence:.1%}")

    st.progress(confidence)

    duration = len(audio) / sr
    st.caption(
        f"Audio duration: {duration:.1f}s · Sample rate: {sr} Hz · "
        f"Raw score: {prediction:.4f}"
    )


if __name__ == "__main__":
    main()

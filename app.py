import streamlit as st
import whisper
import torch
from transformers import pipeline

# Load whisper + bert models
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    emotion_model = pipeline("text-classification", model="bhadresh-savani/bert-base-goemotions-original", top_k=3)
    return whisper_model, emotion_model

st.title("Within - Emotional Awareness")
st.write("Upload a voice note")

audio_file = st.file_uploader("Upload audio (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"])

if audio_file:
    whisper_model, emotion_model = load_models()
    
    with st.spinner("Transcribing..."):
        audio_path = f"temp_audio.{audio_file.name.split('.')[-1]}"
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())
        result = whisper_model.transcribe(audio_path)
        transcription = result["text"]
        st.subheader("Transcription")
        st.write(transcription)

    with st.spinner("Analyzing emotions..."):
        emotions = emotion_model(transcription)
        st.subheader("Detected Emotions")
        for e in emotions[0]:
            st.write(f"- **{e['label']}** ({round(e['score'] * 100, 1)}%)")

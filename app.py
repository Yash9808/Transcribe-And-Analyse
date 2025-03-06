import os
import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import T5Tokenizer, T5ForConditionalGeneration
import librosa.display
import whisper
from collections import Counter
from wordcloud import WordCloud
import torch
import soundfile as sf  # For reading audio files
from pydub import AudioSegment  # For converting MP3 to WAV

# Load T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def analyze_sentiment_t5(text):
    """Analyzes sentiment using the T5 model."""
    input_text = f"sst2 sentence: {text}"  # Formatting input for T5 model
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids)
    sentiment = tokenizer.decode(output[0], skip_special_tokens=True)
    return "POSITIVE" if "positive" in sentiment.lower() else "NEGATIVE"

# Load Whisper model
whisper_model = whisper.load_model("base")

# Streamlit UI
st.title("🎤 Audio Sentiment & Feature Analysis")
st.write("Upload an audio file to analyze its sentiment and audio features.")

uploaded_file = st.file_uploader("Choose an Audio File", type=["mp3", "wav"])

if uploaded_file:
    file_path = f"temp/{uploaded_file.name}"
    os.makedirs("temp", exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert MP3 to WAV if it's an MP3 file (needed if you are working with .mp3 files)
    if uploaded_file.type == "audio/mpeg":
        # Use pydub to convert MP3 to WAV
        audio = AudioSegment.from_mp3(file_path)
        wav_path = file_path.replace(".mp3", ".wav")
        audio.export(wav_path, format="wav")
        file_path = wav_path  # Update to the newly created WAV file

    # Load audio using librosa (works for .wav files)
    y, sr = librosa.load(file_path, sr=None)

    # Extract MFCCs (Mel Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Transcribe with Whisper
    result = whisper_model.transcribe(file_path)
    transcribed_text = result["text"]

    # Analyze sentiment
    sentiment = analyze_sentiment_t5(transcribed_text)
    sentiment_color = "green" if sentiment == "POSITIVE" else "red"

    # Display results
    st.subheader("📊 Sentiment Analysis Result")
    st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color}; font-size:20px;'>{sentiment}</span>", unsafe_allow_html=True)

    # Display full transcription
    st.subheader("📝 Full Transcription")
    st.write(transcribed_text)

    # 1️⃣ MFCC Heatmap
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(mfccs, cmap="coolwarm", xticklabels=False, yticklabels=False)
    ax.set_title("MFCC Heatmap")
    st.pyplot(fig)

    # 2️⃣ Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(transcribed_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud of Transcription")
    st.pyplot(fig)

    # Clean up temp files
    os.remove(file_path)

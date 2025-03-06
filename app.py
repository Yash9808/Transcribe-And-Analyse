import os
import streamlit as st
from pydub import AudioSegment
import whisper
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import T5Tokenizer, T5ForConditionalGeneration
from wordcloud import WordCloud

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
st.write("Upload an MP3 file to analyze its sentiment and audio features.")

uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])

if uploaded_file:
    file_path = f"temp/{uploaded_file.name}"
    os.makedirs("temp", exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert MP3 to WAV
    audio = AudioSegment.from_mp3(file_path)
    wav_path = file_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")

    # Load audio
    y, sr = librosa.load(wav_path, sr=None)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Transcribe with Whisper
    result = whisper_model.transcribe(wav_path)
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
    os.remove(wav_path)
    os.remove(file_path)

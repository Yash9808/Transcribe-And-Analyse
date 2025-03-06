import os
import subprocess  # Import subprocess module
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import T5Tokenizer, T5ForConditionalGeneration
import soundfile as sf  # To read audio files without audioread
import streamlit as st
import whisper
from collections import Counter
from wordcloud import WordCloud
import torch

# Function to install ffmpeg (if needed)
def install_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ffmpeg is installed correctly.")
    except subprocess.CalledProcessError:
        print("FFmpeg not found. Installing FFmpeg...")
        os.system('apt-get update')
        os.system('apt-get install -y ffmpeg')
        os.system('apt-get install -y ffprobe')  # Install ffprobe if needed

# Ensure ffmpeg is installed before processing
install_ffmpeg()

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

    # Read MP3 file using soundfile (instead of pydub)
    # Convert MP3 to WAV using librosa (soundfile can also read WAV)
    try:
        y, sr = librosa.load(file_path, sr=None)  # Automatically loads as WAV if MP3 is not provided
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
    
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

    # Extract MFCCs using librosa
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # 1️⃣ MFCC Plot
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

    # 3️⃣ Positive and Negative Word Analysis
    positive_words = []
    negative_words = []
    word_scores = []

    # Sample word list and sentiment scoring (you can implement or use a sentiment analyzer)
    words = transcribed_text.split()
    for word in words:
        sentiment = analyze_sentiment_t5(word)
        word_scores.append((word, sentiment))
        if sentiment == "POSITIVE":
            positive_words.append(word)
        else:
            negative_words.append(word)
    
    # Create a scatter plot for word scores
    positive_words = [score[0] for score in word_scores if score[1] == "POSITIVE"]
    negative_words = [score[0] for score in word_scores if score[1] == "NEGATIVE"]
    
    # Plotting Positive and Negative Words
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(range(len(positive_words)), [1] * len(positive_words), color="green", label="Positive Words")
    ax.scatter(range(len(negative_words)), [0] * len(negative_words), color="red", label="Negative Words")
    ax.set_xlabel("Words")
    ax.set_ylabel("Sentiment")
    ax.set_title("Positive and Negative Word Analysis")
    ax.legend()
    st.pyplot(fig)

    # Clean up temp files
    os.remove(file_path)

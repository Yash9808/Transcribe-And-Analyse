import os
import time
import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import T5Tokenizer, T5ForConditionalGeneration
import librosa.display
import speech_recognition as sr
from wordcloud import WordCloud
import torch
import soundfile as sf  # For reading audio files

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

# Streamlit UI
st.title("🎤 Audio Sentiment & Feature Analysis")
st.write("Upload an audio file to analyze its sentiment and audio features.")

uploaded_file = st.file_uploader("Choose an Audio File", type=["mp3", "wav"])

if uploaded_file:
    file_path = f"temp/{uploaded_file.name}"
    os.makedirs("temp", exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert MP3 to WAV if necessary
    if uploaded_file.type == "audio/mpeg":
        try:
            # Load MP3 directly using librosa (automatically handles MP3 to WAV conversion)
            y, sampling_rate = librosa.load(file_path, sr=None)  # Librosa handles MP3 to WAV conversion
            wav_path = file_path.replace(".mp3", ".wav")
            sf.write(wav_path, y, sampling_rate)  # Save as WAV
            file_path = wav_path  # Update path to the WAV file
        except Exception as e:
            st.error(f"Error converting MP3 to WAV: {str(e)}")

    # Ensure file exists before loading
    time.sleep(1)  # Give time to save file
    if not os.path.exists(file_path):
        st.error("Error: File not found! Please re-upload.")
    else:
        y, sampling_rate = librosa.load(file_path, sr=None)  # Rename sr to avoid conflicts

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sampling_rate, n_mfcc=13)

        # Use SpeechRecognition for transcription
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as audio_file:
            audio_data = recognizer.record(audio_file)
            try:
                transcribed_text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                transcribed_text = "Sorry, I could not understand the audio."
            except sr.RequestError:
                transcribed_text = "Sorry, there was an error with the API."
            except Exception as e:
                transcribed_text = f"An unexpected error occurred: {str(e)}"

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

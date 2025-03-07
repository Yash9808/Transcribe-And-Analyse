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

#import streamlit as st
#import librosa
#import numpy as np
#import matplotlib.pyplot as plt
#from pydub import AudioSegment
#from transformers import T5Tokenizer, T5ForConditionalGeneration
#import os
#import whisper
#from collections import Counter
#import torch

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

def highlight_words(text, sentiment="POSITIVE"):
    """Highlight positive and negative words in transcription."""
    # Create a list of positive and negative words (simple example, expand as needed)
    positive_words = {"good", "great", "awesome", "happy", "positive", "love"}
    negative_words = {"bad", "sad", "angry", "negative", "hate", "awful"}
    
    # Split the transcription into words
    words = text.split()
    
    highlighted_text = []
    
    for word in words:
        if word.lower() in positive_words:
            highlighted_text.append(f"<span style='color:green'>{word}</span>")  # Green for positive
        elif word.lower() in negative_words:
            highlighted_text.append(f"<span style='color:red'>{word}</span>")  # Red for negative
        else:
            highlighted_text.append(word)  # Leave neutral words unchanged
    
    # Join the words back into a string
    return ' '.join(highlighted_text)

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
    
    # Get audio length in seconds
    audio_length = librosa.get_duration(y=y, sr=sr)

    # Transcribe with Whisper
    result = whisper_model.transcribe(wav_path)
    transcribed_text = result["text"]

    # Analyze sentiment
    sentiment = analyze_sentiment_t5(transcribed_text)
    sentiment_color = "green" if sentiment == "POSITIVE" else "red"

    # Highlight positive and negative words in transcription
    highlighted_transcription = highlight_words(transcribed_text, sentiment)

    # Display results
    st.subheader("📊 Sentiment Analysis Result")
    st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color}; font-size:20px;'>{sentiment}</span>", unsafe_allow_html=True)
    
    # Display full transcription with highlighted words
    st.subheader("📝 Full Transcription")
    st.markdown(highlighted_transcription, unsafe_allow_html=True)

    # Plot sentiment score vs. audio length
    fig, ax = plt.subplots(figsize=(10, 5))
    sentiment_score = 1 if sentiment == "POSITIVE" else 0  # Simplified sentiment score: 1 for POSITIVE, 0 for NEGATIVE
    ax.barh(["Sentiment"], [sentiment_score], color=sentiment_color)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Sentiment Score")
    ax.set_title(f"Sentiment Score vs. Audio Length (Duration: {audio_length:.2f} seconds)")
    st.pyplot(fig)

    # Clean up temp files
    os.remove(wav_path)
    os.remove(file_path)



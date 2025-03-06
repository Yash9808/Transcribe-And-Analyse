import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import whisper
from collections import Counter
import torch

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

    # Transcribe with Whisper
    result = whisper_model.transcribe(wav_path)
    transcribed_text = result["text"]

    # Analyze sentiment word by word
    words = transcribed_text.split()
    word_sentiments = []
    for word in words:
        sentiment = analyze_sentiment_t5(word)
        word_sentiments.append((word, sentiment))

    # Plotting the words sentiment
    positive_words = [word for word, sentiment in word_sentiments if sentiment == "POSITIVE"]
    negative_words = [word for word, sentiment in word_sentiments if sentiment == "NEGATIVE"]
    
    # Scatter plot for word sentiment
    plt.figure(figsize=(10, 5))
    positive_scores = [i for i, (word, sentiment) in enumerate(word_sentiments) if sentiment == "POSITIVE"]
    negative_scores = [i for i, (word, sentiment) in enumerate(word_sentiments) if sentiment == "NEGATIVE"]

    plt.scatter(positive_scores, [1] * len(positive_scores), color='green', label="Positive", s=100)
    plt.scatter(negative_scores, [1] * len(negative_scores), color='red', label="Negative", s=100)

    plt.yticks([])  # Hide y-axis
    plt.xlabel("Words")
    plt.title("Word Sentiment Scatter Plot")
    plt.legend()
    st.pyplot(plt)

    # Highlight positive and negative words in transcription
    highlighted_text = ""
    for word, sentiment in word_sentiments:
        if sentiment == "POSITIVE":
            highlighted_text += f"<span style='color:green'>{word}</span> "
        else:
            highlighted_text += f"<span style='color:red'>{word}</span> "

    st.subheader("📝 Transcription with Sentiment Highlighting")
    st.markdown(highlighted_text, unsafe_allow_html=True)

    # Calculate the length of audio
    audio_length = librosa.get_duration(y=y, sr=sr)
    st.write(f"⏳ Audio Length: {audio_length:.2f} seconds")

    # Calculate and display overall sentiment (for the whole transcription)
    overall_sentiment = analyze_sentiment_t5(transcribed_text)
    overall_sentiment_color = "green" if overall_sentiment == "POSITIVE" else "red"
    st.subheader("📊 Overall Sentiment Analysis Result")
    st.markdown(f"**Overall Sentiment:** <span style='color:{overall_sentiment_color}; font-size:20px;'>{overall_sentiment}</span>", unsafe_allow_html=True)

    # Clean up temp files
    os.remove(wav_path)
    os.remove(file_path)

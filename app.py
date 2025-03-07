import os
import time
import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5ForConditionalGeneration
import librosa.display
import speech_recognition as sr
import soundfile as sf  # For reading audio files

# Load T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Define positive and negative words
POSITIVE_WORDS = {"good", "great", "excellent", "awesome", "happy", "love", "positive", "satisfied"}
NEGATIVE_WORDS = {"bad", "terrible", "awful", "sad", "angry", "negative", "hate", "problem"}

def analyze_sentiment_t5(text):
    """Analyzes sentiment using the T5 model."""
    input_text = f"sst2 sentence: {text}"  # Formatting input for T5 model
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids)
    sentiment = tokenizer.decode(output[0], skip_special_tokens=True)
    return "POSITIVE" if "positive" in sentiment.lower() else "NEGATIVE"

def highlight_words(text):
    """Highlights positive words in green and negative words in red."""
    words = text.split()
    highlighted_text = []

    for word in words:
        if word.lower() in POSITIVE_WORDS:
            highlighted_text.append(f"<span style='color:green'>{word}</span>")  # Green for positive words
        elif word.lower() in NEGATIVE_WORDS:
            highlighted_text.append(f"<span style='color:red'>{word}</span>")  # Red for negative words
        else:
            highlighted_text.append(word)  # Leave neutral words unchanged

    return ' '.join(highlighted_text)

def extract_agent_question_and_problem(text):
    """Extracts the agent's question and customer's problem from the transcribed text."""
    sentences = text.split(". ")  # Splitting by sentence
    agent_question = ""
    problem_statement = ""

    for sentence in sentences:
        if "?" in sentence:
            agent_question = sentence.strip()  # Capture the first question found
            break

    for sentence in sentences:
        for word in NEGATIVE_WORDS:
            if word in sentence.lower():
                problem_statement = sentence.strip()
                break

    return agent_question, problem_statement

# Streamlit UI
st.title("🎤 ✍ Transcribe and Analyze")
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
            y, sampling_rate = librosa.load(file_path, sr=None)
            wav_path = file_path.replace(".mp3", ".wav")
            sf.write(wav_path, y, sampling_rate)  # Save as WAV
            file_path = wav_path  # Update path to the WAV file
        except Exception as e:
            st.error(f"Error converting MP3 to WAV: {str(e)}")

    # Ensure file exists before loading
    time.sleep(1)
    if not os.path.exists(file_path):
        st.error("Error: File not found! Please re-upload.")
    else:
        y, sampling_rate = librosa.load(file_path, sr=None)
        audio_length = librosa.get_duration(y=y, sr=sampling_rate)

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

        # Extract Agent Question & Customer Problem
        agent_question, problem_statement = extract_agent_question_and_problem(transcribed_text)

        # Display results
        st.subheader("📊 Sentiment Analysis Result")
        st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color}; font-size:20px;'>{sentiment}</span>", unsafe_allow_html=True)

        # Display transcription with highlights
        st.subheader("📝 Full Transcription")
        highlighted_text = highlight_words(transcribed_text)

        # Create a two-column layout
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(highlighted_text, unsafe_allow_html=True)

        with col2:
            st.subheader("📌 Agent's Question")
            st.info(agent_question if agent_question else "No question detected.")

            st.subheader("⚠ Problem Described")
            st.warning(problem_statement if problem_statement else "No issue detected.")

        # Sentiment Word Plot
        words = transcribed_text.split()
        positive_counts = [i for i, word in enumerate(words) if word.lower() in POSITIVE_WORDS]
        negative_counts = [i for i, word in enumerate(words) if word.lower() in NEGATIVE_WORDS]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(positive_counts, [1] * len(positive_counts), color="green", label="Positive Words")
        ax.scatter(negative_counts, [0] * len(negative_counts), color="red", label="Negative Words")
        ax.plot(positive_counts, [1] * len(positive_counts), color="green", linestyle="dotted", alpha=0.5)
        ax.plot(negative_counts, [0] * len(negative_counts), color="red", linestyle="dotted", alpha=0.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Negative", "Positive"])
        ax.set_xlabel("Word Position in Transcription")
        ax.set_title(f"Positive & Negative Word Distribution Over Audio Length ({audio_length:.2f} sec)")
        ax.legend()
        st.pyplot(fig)

        # Clean up temp files
        os.remove(file_path)

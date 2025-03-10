import os
import time
import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import speech_recognition as sr
import soundfile as sf  # For audio file handling

# Define positive and negative words for sentiment analysis
POSITIVE_WORDS = {"good", "great", "excellent", "awesome", "happy", "love", "positive", "satisfied"}
NEGATIVE_WORDS = {"bad", "terrible", "awful", "sad", "angry", "negative", "hate", "problem", "not good service", "Rubbish", "frustrating"}

def analyze_sentiment(text):
    """Analyzes sentiment by identifying positive and negative words."""
    words = text.split()
    positive_count = sum(1 for word in words if word.lower() in POSITIVE_WORDS)
    negative_count = sum(1 for word in words if word.lower() in NEGATIVE_WORDS)
    
    if positive_count > negative_count:
        return "POSITIVE"
    elif negative_count > positive_count:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def highlight_words(text):
    """Highlights positive words in green and negative words in red."""
    words = text.split()
    highlighted_text = []

    for word in words:
        if word.lower() in POSITIVE_WORDS:
            highlighted_text.append(f"<span style='color:green'>{word}</span>")
        elif word.lower() in NEGATIVE_WORDS:
            highlighted_text.append(f"<span style='color:red'>{word}</span>")
        else:
            highlighted_text.append(word)

    return ' '.join(highlighted_text)

def extract_agent_question_and_problem(text):
    """Extracts the agent's question and customer's problem from the transcribed text."""
    sentences = text.split(". ")
    agent_question = ""
    problem_statement = ""

    # Loop through sentences to find the first question
    for sentence in sentences:
        if "?" in sentence:
            agent_question = sentence.strip()
            break

    # Loop through sentences to find a problem statement based on negative words
    for sentence in sentences:
        if any(word in sentence.lower() for word in NEGATIVE_WORDS):
            problem_statement = sentence.strip()
            break

    # If no problem statement found, search for more descriptive words
    if not problem_statement:
        for sentence in sentences:
            if len(sentence.split()) > 5:  # Assume problems are described in slightly longer sentences
                problem_statement = sentence.strip()
                break

    return agent_question, problem_statement

def transcribe_audio(file_path):
    """Transcribes audio using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as audio_file:
        audio_data = recognizer.record(audio_file)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Sorry, there was an error with the API."
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

# Streamlit UI
st.title("🎤 ✍ Transcribe and Analyze")
st.write("Upload an audio file to analyze its sentiment and audio features.")

uploaded_file = st.file_uploader("Choose an Audio File", type=["mp3", "wav"])

if uploaded_file:
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert MP3 to WAV if necessary
    if uploaded_file.type == "audio/mpeg":
        try:
            y, sampling_rate = librosa.load(file_path, sr=None)
            wav_path = file_path.replace(".mp3", ".wav")
            sf.write(wav_path, y, sampling_rate)
            file_path = wav_path  # Use converted WAV file
        except Exception as e:
            st.error(f"Error converting MP3 to WAV: {str(e)}")

    # Ensure file exists
    time.sleep(1)
    if not os.path.exists(file_path):
        st.error("Error: File not found! Please re-upload.")
    else:
        # Load audio and get duration
        y, sampling_rate = librosa.load(file_path, sr=None)
        audio_length = librosa.get_duration(y=y, sr=sampling_rate)

        # Transcribe the audio
        transcribed_text = transcribe_audio(file_path)

        # Debugging output
        print("Transcribed Text:", transcribed_text)

        # Analyze sentiment
        sentiment = analyze_sentiment(transcribed_text)
        sentiment_color = "green" if sentiment == "POSITIVE" else "red"

        # Extract Agent Question & Customer Problem
        agent_question, problem_statement = extract_agent_question_and_problem(transcribed_text)

        # Display results
        st.subheader("📊 Sentiment Analysis Result")
        st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color}; font-size:20px;'>{sentiment}</span>", unsafe_allow_html=True)

        # Display transcription with highlights
        st.subheader("📝 Full Transcription")
        highlighted_text = highlight_words(transcribed_text)

        # Modify to make transcription area wider
        st.markdown(
            f"<div style='width: 100%; border: 1px solid #ddd; padding: 20px; border-radius: 5px; font-size: 16px;'>{highlighted_text}</div>", 
            unsafe_allow_html=True
        )

        # Sentiment Word Plot
        words = transcribed_text.split()
        word_positions = np.linspace(0, audio_length, len(words))  # Map words to time
        positive_counts = [word_positions[i] for i, word in enumerate(words) if word.lower() in POSITIVE_WORDS]
        negative_counts = [word_positions[i] for i, word in enumerate(words) if word.lower() in NEGATIVE_WORDS]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(positive_counts, [1] * len(positive_counts), color="green", label="Positive Words")
        ax.scatter(negative_counts, [0] * len(negative_counts), color="red", label="Negative Words")
        ax.plot(positive_counts, [1] * len(positive_counts), color="green", linestyle="dotted", alpha=0.5)
        ax.plot(negative_counts, [0] * len(negative_counts), color="red", linestyle="dotted", alpha=0.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Negative", "Positive"])
        ax.set_xlabel("Time in Seconds")
        ax.set_title(f"Positive & Negative Word Distribution Over Audio Length ({audio_length:.2f} sec)")
        ax.legend()
        st.pyplot(fig)

        # Move the "Agent's Question" and "Problem Described" sections here
        st.subheader("📌 Agent's Question")
        st.info(agent_question if agent_question else "No question detected.")

        st.subheader("⚠ Problem Described")
        st.warning(problem_statement if problem_statement else "No issue detected.")

        # Clean up temp files
        os.remove(file_path)

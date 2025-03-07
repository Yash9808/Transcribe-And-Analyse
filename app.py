import os
import time
import streamlit as st
import speech_recognition as sr
import soundfile as sf
from pydub import AudioSegment  # ✅ Fix for MP3 to WAV conversion
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

POSITIVE_WORDS = {"good", "great", "excellent", "awesome", "happy", "love", "positive", "satisfied"}
NEGATIVE_WORDS = {"bad", "terrible", "awful", "sad", "angry", "negative", "hate", "problem"}

def analyze_sentiment_t5(text):
    """Analyzes sentiment using the T5 model."""
    input_text = f"sst2 sentence: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids)
    sentiment = tokenizer.decode(output[0], skip_special_tokens=True)
    return "POSITIVE" if "positive" in sentiment.lower() else "NEGATIVE"

def convert_audio_to_wav(file_path):
    """Converts MP3 to WAV using pydub if needed."""
    if file_path.endswith(".mp3"):
        audio = AudioSegment.from_mp3(file_path)
        wav_path = file_path.replace(".mp3", ".wav")
        audio.export(wav_path, format="wav")
        return wav_path
    return file_path

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
            return f"Error: {str(e)}"

# Streamlit UI
st.title("🎤 ✍ Transcribe and Analyze")
st.write("Upload an audio file to analyze its sentiment and audio features.")

uploaded_file = st.file_uploader("Choose an Audio File", type=["mp3", "wav"])

if uploaded_file:
    file_path = f"temp/{uploaded_file.name}"
    os.makedirs("temp", exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert audio if needed
    file_path = convert_audio_to_wav(file_path)

    time.sleep(1)
    if not os.path.exists(file_path):
        st.error("Error: File not found! Please re-upload.")
    else:
        # **✅ Fix: Use correct WAV file for transcription**
        transcribed_text = transcribe_audio(file_path)

        # **Debugging output** (for logs)
        print("Transcribed Text:", transcribed_text)

        # **Check if transcription is valid**
        if transcribed_text.isdigit() or len(transcribed_text) < 5:
            st.error("Transcription failed. Try a clearer audio file.")
        else:
            # Analyze sentiment
            sentiment = analyze_sentiment_t5(transcribed_text)
            sentiment_color = "green" if sentiment == "POSITIVE" else "red"

            # Display results
            st.subheader("📊 Sentiment Analysis Result")
            st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color}; font-size:20px;'>{sentiment}</span>", unsafe_allow_html=True)

            # **Display transcription in a large text box**
            st.subheader("📝 Full Transcription")
            st.text_area("Transcription:", value=transcribed_text, height=200)

        # Clean up temp files
        os.remove(file_path)

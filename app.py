import os
os.system('apt-get update')
os.system('apt-get install -y ffmpeg')
os.system('apt-get install -y ffprobe')
import streamlit as st
import librosa
from transformers import T5Tokenizer, T5ForConditionalGeneration
import soundfile as sf  # Use soundfile to read audio
import os
import whisper
from pydub import AudioSegment  # For converting MP3 to WAV

# Load T5 model and tokenizer for sentiment analysis
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

    # Convert MP3 to WAV (using pydub)
    wav_path = file_path.replace(".mp3", ".wav")
    audio = AudioSegment.from_mp3(file_path)  # Convert MP3 to WAV
    audio.export(wav_path, format="wav")     # Export as WAV file

    # Load audio using soundfile
    y, sr = sf.read(wav_path)

    # Transcribe with Whisper
    result = whisper_model.transcribe(wav_path)
    transcribed_text = result["text"]

    # Analyze sentiment using the T5 model
    sentiment = analyze_sentiment_t5(transcribed_text)
    sentiment_color = "green" if sentiment == "POSITIVE" else "red"

    # Display results
    st.subheader("📊 Sentiment Analysis Result")
    st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color}; font-size:20px;'>{sentiment}</span>", unsafe_allow_html=True)

    # Display full transcription
    st.subheader("📝 Full Transcription")
    st.write(transcribed_text)

    # Calculate the length of audio
    audio_length = librosa.get_duration(y=y, sr=sr)
    st.write(f"⏳ Audio Length: {audio_length:.2f} seconds")

    # Clean up temp files
    os.remove(wav_path)
    os.remove(file_path)

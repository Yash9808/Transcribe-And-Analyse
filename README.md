# 🎤 Audio Transcription and Sentiment Analysis

## 📌 Overview
This project is a Streamlit-based web application that allows users to upload an audio file, transcribe its content, analyze sentiment, and visualize key insights such as word distribution over time. It is particularly useful for customer service reviews, feedback analysis, and sentiment tracking.

## 🚀 Features
- **Audio Transcription**: Converts speech to text using the Google Speech Recognition API.
- **Sentiment Analysis**: Identifies positive and negative words within the transcription.
- **Keyword Highlighting**: Highlights positive words in green and negative words in red.
- **Question and Problem Extraction**: Detects agent questions and customer complaints.
- **Word Distribution Visualization**: Displays positive and negative words plotted over the duration of the audio file.
- **MP3 to WAV Conversion**: Automatically converts MP3 files to WAV for compatibility.

## 🛠 Technologies Used
- **Python**
- **Streamlit** (for web UI)
- **Librosa** (for audio processing)
- **SpeechRecognition** (for transcription)
- **Matplotlib & NumPy** (for visualization)
- **Soundfile** (for file conversion)

## 📥 Installation & Setup

### Prerequisites
Ensure you have Python installed. You can install dependencies using:
```bash
pip install streamlit librosa speechrecognition matplotlib numpy soundfile
```

### Running the Application
Run the Streamlit app with:
```bash
streamlit run app.py
```

## 📌 How to Use
1. **Upload an Audio File**: Select an MP3 or WAV file to process.
2. **Automatic Transcription**: The app will transcribe the speech into text.
3. **Sentiment Analysis**: The app will analyze and highlight sentiment in the text.
4. **Word Distribution Visualization**: View the spread of positive and negative words over time.
5. **Agent & Problem Extraction**: Identify the key agent questions and customer issues.

## 🎯 Use Cases
- **Customer Support Analysis**
- **Podcast & Meeting Transcriptions**
- **Interview Analysis**
- **Sentiment Tracking in Conversations**

## 🤖 Future Improvements
- Support for more languages
- Enhanced NLP processing for better sentiment classification
- Cloud storage for storing transcriptions

App link: https://yp2opqdw5du4x4cevfryet.streamlit.app/
---

Happy Transcribing! 🎙️📊


import streamlit as st
import os
import tempfile
import yt_dlp
from modules.audio_processing import extract_audio, filter_silence
from modules.transcription import transcribe_audio, detect_language
from modules.summarization import summarize_text
from modules.question_generation import generate_questions
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit UI
st.title("Real-Time Long-Form Video Processing App")
st.write("Upload a video or provide a YouTube URL to transcribe, summarize, and generate questions.")

# Input options
input_option = st.radio("Choose input method:", ("Upload Video", "YouTube URL"))
video_path = None
audio_path = None
filtered_audio_path = None

def update_progress(d, progress_text, progress_bar):
    if d['status'] == 'downloading':
        percent = d.get('downloaded_bytes', 0) / d.get('total_bytes', 1) * 100
        progress_bar.progress(int(percent))
        progress_text.text(f"Downloading... {percent:.1f}%")
    elif d['status'] == 'finished':
        progress_text.text("Download complete!")
        progress_bar.progress(100)

if input_option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mkv", "avi"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        st.write("Video uploaded successfully!")
else:
    youtube_url = st.text_input("Enter YouTube URL:")
    if youtube_url:
        progress_text = st.empty()
        progress_bar = st.progress(0)

        ydl_opts = {
            'format': 'bestaudio[ext=m4a]',
            'outtmpl': 'downloads/%(id)s.%(ext)s',
            'progress_hooks': [lambda d: update_progress(d, progress_text, progress_bar)],
            'quiet': True,
            'no_warnings': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                audio_path = ydl.prepare_filename(info)
                logger.info("Audio downloaded successfully: %s", audio_path)
                st.write("Audio downloaded successfully!")
        except Exception as e:
            logger.error("Failed to download audio: %s", str(e))
            st.error(f"Error downloading audio: {str(e)}")
            audio_path = None

# Process the video or audio
if video_path:
    audio_path = "temp_audio.wav"
    try:
        extract_audio(video_path, audio_path)
        st.write("Audio extracted successfully!")
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        audio_path = None
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

# Proceed with processing if audio is available
if audio_path:
    filtered_audio_path = "filtered_audio.wav"
    try:
        filter_silence(audio_path, filtered_audio_path)
        st.write("Audio filtered successfully!")
    except Exception as e:
        st.error(f"Error filtering audio: {str(e)}")
        filtered_audio_path = None
    finally:
        if audio_path != filtered_audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

    if filtered_audio_path:
        # Detect language
        st.subheader("Language Detection")
        try:
            language = detect_language(filtered_audio_path)
            st.write(f"Detected language: {language}")
        except Exception as e:
            st.error(f"Error detecting language: {str(e)}")
            language = "en"
            st.write("Defaulting to English for transcription.")

        # Real-time transcription
        st.subheader("Real-Time Transcription")
        transcription_area = st.container()
        with transcription_area:
            transcription = transcribe_audio(filtered_audio_path, st, language=language)
        if transcription:
            st.write("Transcription completed!")
        else:
            st.error("Transcription failed.")

        # Summarization
        st.subheader("Summary")
        if transcription:
            try:
                summary = summarize_text(transcription)
                st.write(summary)
            except Exception as e:
                st.error(f"Error summarizing text: {str(e)}")

        # Question Generation
        st.subheader("Generated Questions")
        if transcription:
            try:
                questions = generate_questions(transcription)
                for q in questions:
                    st.write(f"- {q}")
            except Exception as e:
                st.error(f"Error generating questions: {str(e)}")

    # Clean up
    if filtered_audio_path and os.path.exists(filtered_audio_path):
        os.remove(filtered_audio_path)
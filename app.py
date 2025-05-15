import streamlit as st
import os
import tempfile
import yt_dlp
from modules.audio_processing import extract_audio, filter_silence
from modules.transcription import transcribe_audio, detect_language_confidence
from modules.summarization import summarize_text
from modules.question_generation import generate_questions
from modules.hindi_support import clean_hindi_text
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for download progress
if 'download_status' not in st.session_state:
    st.session_state.download_status = None
if 'current_video_path' not in st.session_state:
    st.session_state.current_video_path = None

def format_bytes(bytes):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} TB"

def download_with_progress(url, progress_text, progress_bar):
    """Download video with progress tracking and error handling."""
    try:
        # Create downloads directory if it doesn't exist
        Path('downloads').mkdir(exist_ok=True)
        
        def progress_hook(d):
            if d['status'] == 'downloading':
                try:
                    total = d.get('total_bytes', 0) or d.get('total_bytes_estimate', 0)
                    downloaded = d.get('downloaded_bytes', 0)
                    speed = d.get('speed', 0)
                    
                    if total > 0:
                        percent = (downloaded / total) * 100
                        progress_bar.progress(int(percent))
                        
                        # Format progress message
                        speed_str = f"{format_bytes(speed)}/s" if speed else "..."
                        progress_text.text(
                            f"Downloading... {percent:.1f}% "
                            f"({format_bytes(downloaded)} / {format_bytes(total)}) "
                            f"at {speed_str}"
                        )
                except Exception as e:
                    progress_text.text(f"Downloading... (Progress calculation failed: {str(e)})")
                    
            elif d['status'] == 'finished':
                progress_text.text("Download complete! Processing video...")
                progress_bar.progress(100)
                
            elif d['status'] == 'error':
                progress_text.error(f"Download error: {d.get('error', 'Unknown error')}")

        ydl_opts = {
            'format': 'best',  # Download best quality
            'outtmpl': 'downloads/%(id)s.%(ext)s',
            'progress_hooks': [progress_hook],
            'quiet': True,
            'no_warnings': True,
            'retries': 10,  # Retry up to 10 times
            'fragment_retries': 10,
            'ignoreerrors': False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            logger.info(f"Video downloaded successfully: {video_path}")
            return video_path

    except Exception as e:
        error_msg = str(e)
        if "HTTP Error 429" in error_msg:
            progress_text.error("Download rate limited. Please try again later.")
        elif "No space left on device" in error_msg:
            progress_text.error("Not enough disk space for download.")
        else:
            progress_text.error(f"Download failed: {error_msg}")
        return None

# Streamlit UI setup
st.title("Advanced Video Transcription & Analysis")
st.write("Upload a video or provide a YouTube URL for transcription, summarization, and analysis. Supports English, Hindi, and Hinglish content.")

# Language selection
language_options = {
    "Auto-detect": None,
    "English": "en",
    "Hindi": "hi",
    "Hinglish (Hindi + English)": "hi-en"
}

selected_language = st.selectbox(
    "Select content language (or auto-detect)",
    options=list(language_options.keys())
)

# Input options
input_option = st.radio("Choose input method:", ("Upload Video", "YouTube URL"))
video_path = None
audio_path = None
filtered_audio_path = None

# File handling based on input method
if input_option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mkv", "avi", "mov"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        st.success("Video uploaded successfully!")
else:
    youtube_url = st.text_input("Enter YouTube URL:")
    if youtube_url and youtube_url != st.session_state.get('last_url', ''):
        st.session_state.last_url = youtube_url
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        video_path = download_with_progress(youtube_url, progress_text, progress_bar)
        if video_path:
            st.success("Video downloaded successfully!")
            st.session_state.current_video_path = video_path
        else:
            st.error("Video download failed.")
            st.stop()

# Process the video
if video_path:
    try:
        # Set up audio paths
        video_path_obj = Path(video_path)
        audio_path = str(video_path_obj.parent / f"{video_path_obj.stem}_audio.wav")
        filtered_audio_path = str(video_path_obj.parent / f"{video_path_obj.stem}_filtered.wav")
        
        # Extract and process audio
        with st.spinner("Extracting audio..."):
            extract_audio(video_path, audio_path)
            st.success("Audio extracted successfully!")
        
        with st.spinner("Processing audio..."):
            filter_silence(audio_path, filtered_audio_path)
            st.success("Audio processed successfully!")
        
        # Language detection and transcription
        st.subheader("Content Analysis")
        try:
            language = language_options[selected_language]
            if language is None:  # Auto-detect
                language, is_mixed, confidence = detect_language_confidence(filtered_audio_path)
                language_display = "Hinglish" if is_mixed else language.upper()
                st.info(f"Detected language: {language_display} (confidence: {confidence:.2f})")
            else:
                is_mixed = language == "hi-en"
                if is_mixed:
                    language = "hi"
                st.info(f"Using selected language: {selected_language}")
                
            # Transcription
            st.subheader("Transcription")
            transcription_container = st.container()
            with transcription_container:
                transcription = transcribe_audio(
                    filtered_audio_path,
                    transcription_area=transcription_container,
                    language=language
                )
                
            if transcription:
                st.success("Transcription completed!")
                
                # Clean Hindi/Hinglish text if needed
                if language == "hi" or is_mixed:
                    transcription = clean_hindi_text(transcription)
                
                # Show translation option for Hindi content
                if language == "hi" and not is_mixed:
                    if st.checkbox("Show English translation"):
                        with st.spinner("Translating..."):
                            try:
                                from transformers import pipeline
                                translator = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
                                translation = translator(transcription, max_length=1000)[0]["translation_text"]
                                st.text_area("English Translation", translation, height=200)
                            except Exception as e:
                                st.error(f"Translation error: {str(e)}")

                # Summarization
                st.subheader("Summary")
                with st.spinner("Generating summary..."):
                    try:
                        summary = summarize_text(transcription)
                        st.text_area("Content Summary", summary, height=200)
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")

                # Question Generation
                st.subheader("Key Questions")
                with st.spinner("Generating questions..."):
                    try:
                        questions = generate_questions(transcription)
                        for q in questions:
                            st.write(f"- {q}")
                    except Exception as e:
                        st.error(f"Error generating questions: {str(e)}")
            else:
                st.error("Transcription failed.")
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            
    finally:
        # Clean up files
        for file_path in [video_path, audio_path, filtered_audio_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error removing file {file_path}: {str(e)}")

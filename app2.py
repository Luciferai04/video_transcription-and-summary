import streamlit as st
import os
import tempfile
import re
import time
from urllib.parse import urlparse
import logging
import traceback

# Import custom modules
from modules.transcription import (
    download_youtube_video, 
    transcribe_audio, 
    detect_language, 
    process_audio_chunks
)
from modules.summarization import generate_summary
from modules.question_generation import generate_questions
from modules.hindi_support import clean_hindi_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Video Transcription & Summary",
    page_icon="🎬",
    layout="wide"
)

# App title and description
st.title("Video Transcription & Summary")
st.markdown("""
This app allows you to transcribe videos, generate summaries, and create questions from the content.
Special support is available for Hindi content with Hinglish (romanized Hindi) transcription.
""")

# Function to validate YouTube URL
def is_valid_youtube_url(url):
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    youtube_match = re.match(youtube_regex, url)
    return bool(youtube_match)

# Function to process the video
def process_video(video_path, is_youtube=False):
    try:
        with st.spinner("Detecting language..."):
            # Detect language from a sample of the video
            language_info = detect_language(video_path)
            detected_language = language_info["language"]
            confidence = language_info["confidence"]
            st.success(f"Detected language: {detected_language.capitalize()} (Confidence: {confidence:.2f})")
            
            # Create language-specific prompt for transcription
            prompt = ""
            if detected_language == "hindi":
                prompt = "Transcribe the following audio in Hinglish (romanized Hindi)."
                st.info("Hindi content detected. Will transcribe in Hinglish (romanized Hindi).")
            
        # Process the video in chunks and display progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Define callback for progress updates
        def update_progress(current, total):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"Transcribing: {int(progress * 100)}% complete")
        
        # Transcribe the video
        with st.spinner("Transcribing video..."):
            transcription_results = process_audio_chunks(
                video_path, 
                language=detected_language,
                prompt=prompt,
                progress_callback=update_progress
            )
            
            full_transcription = " ".join([chunk["text"] for chunk in transcription_results])
            
            # Clean Hindi text if detected language is Hindi
            if detected_language == "hindi":
                full_transcription = clean_hindi_text(full_transcription)
        
        progress_bar.progress(1.0)
        status_text.text("Transcription complete!")
        
        # Display the transcription
        st.subheader("Transcription")
        st.text_area("Full Transcription", full_transcription, height=200)
        
        # Generate summary
        with st.spinner("Generating summary..."):
            summary = generate_summary(full_transcription, language=detected_language)
            st.subheader("Summary")
            st.write(summary)
        
        # Generate questions
        with st.spinner("Generating questions..."):
            questions = generate_questions(full_transcription)
            st.subheader("Questions")
            for i, question in enumerate(questions, 1):
                st.write(f"{i}. {question}")
        
        return True
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        logger.error(f"Error processing video: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Main app logic
def main():
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Video", "YouTube URL"])
    
    with tab1:
        st.header("Upload a Video File")
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                video_path = tmp_file.name
            
            st.video(uploaded_file)
            
            if st.button("Process Video", key="process_uploaded"):
                process_video(video_path)
                # Clean up the temporary file
                try:
                    os.unlink(video_path)
                except:
                    pass
    
    with tab2:
        st.header("Enter YouTube URL")
        youtube_url = st.text_input("YouTube URL")
        
        if youtube_url:
            if is_valid_youtube_url(youtube_url):
                if st.button("Process YouTube Video", key="process_youtube"):
                    with st.spinner("Downloading YouTube video..."):
                        try:
                            video_path = download_youtube_video(youtube_url)
                            st.success("YouTube video downloaded successfully!")
                            process_video(video_path, is_youtube=True)
                            # Clean up the downloaded file
                            try:
                                os.unlink(video_path)
                            except:
                                pass
                        except Exception as e:
                            st.error(f"Error downloading YouTube video: {str(e)}")
            else:
                st.error("Invalid YouTube URL. Please enter a valid URL.")

if __name__ == "__main__":
    main()


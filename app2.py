
import streamlit as st
import os
import tempfile
import re
import time
import gc
from urllib.parse import urlparse
import logging
import traceback
import torch

# Import custom modules
from modules.transcription import (
    transcribe_audio, 
    detect_language_confidence
)
from modules.utils import download_youtube_video
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
Enhanced support for English content with improved accuracy and performance.
""")

# Initialize session state for tracking processing steps
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = None
if 'language_detected' not in st.session_state:
    st.session_state.language_detected = None
if 'cleanup_files' not in st.session_state:
    st.session_state.cleanup_files = []

# Function to validate YouTube URL
def is_valid_youtube_url(url):
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    youtube_match = re.match(youtube_regex, url)
    return bool(youtube_match)

# Function to process the video
def summary_progress_callback(progress, message):
    """Callback to update UI during summarization"""
    if 'summary_progress_bar' in st.session_state:
        st.session_state.summary_progress_bar.progress(progress)
    if 'summary_status_text' in st.session_state:
        st.session_state.summary_status_text.text(message)

def cleanup_temp_files():
    """Clean up temporary files"""
    if 'cleanup_files' in st.session_state:
        for file_path in st.session_state.cleanup_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up file {file_path}: {str(e)}")
        st.session_state.cleanup_files = []

def process_video(video_path, is_youtube=False):
    try:
        # Setup memory management
        gc.collect()
        if hasattr(torch, 'cuda'):
            torch.cuda.empty_cache()
        
        # Add file to cleanup list
        if is_youtube:
            st.session_state.cleanup_files.append(video_path)
        
        # Initialize processing status
        st.session_state.processing_status = "detecting_language"
        
        with st.spinner("Detecting language..."):
            # Create status containers
            language_detection_container = st.container()
            language_status = language_detection_container.empty()
            language_status.info("Analyzing audio language...")
            
            # Detect language from a sample of the video
            language_code, is_mixed, confidence = detect_language_confidence(video_path)
            
            # Convert language code to full name
            if language_code == "hi":
                detected_language = "hindi"
            elif language_code == "en":
                detected_language = "english"
            else:
                detected_language = language_code
                
            # Store in session state
            st.session_state.language_detected = detected_language
                
            # Update UI with language detection result
            language_status.success(f"Detected language: {detected_language.capitalize()} (Confidence: {confidence:.2f})")
            
            # Create language-specific prompt for transcription
            prompt = ""
            if detected_language == "hindi":
                prompt = "Transcribe the following audio in Hinglish (romanized Hindi)."
                language_detection_container.info("Hindi content detected. Will transcribe in Hinglish (romanized Hindi).")
            elif detected_language == "english":
                language_detection_container.info("English content detected. Using enhanced English transcription features.")
            
        # Update processing status
        st.session_state.processing_status = "transcribing"
            
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
            # Create a container for the transcription progress
            transcription_area = st.container()
            transcription_status = transcription_area.empty()
            transcription_status.info(f"Starting transcription process for {detected_language} content...")
            
            # Use the transcribe_audio function which already handles chunking
            full_transcription = transcribe_audio(
                video_path,
                transcription_area=transcription_area,
                language=detected_language
            )
            
            if full_transcription is None:
                raise ValueError("Transcription failed. Please try a different video or check the audio quality.")
                
            # Clean Hindi text if detected language is Hindi
            if detected_language == "hindi":
                full_transcription = clean_hindi_text(full_transcription)
        
        progress_bar.progress(1.0)
        status_text.text("Transcription complete!")
        
        # Update processing status
        st.session_state.processing_status = "displaying_transcription"
        
        # Display the transcription
        st.subheader("Transcription")
        st.text_area("Full Transcription", full_transcription, height=200, key="main_transcript")
        
        # Download button for transcription
        if full_transcription:
            st.download_button(
                label="Download Full Transcription",
                data=full_transcription,
                file_name="transcription.txt",
                mime="text/plain"
            )
        
        # Memory cleanup after transcription
        gc.collect()
        if hasattr(torch, 'cuda'):
            torch.cuda.empty_cache()
        
        # Update processing status
        st.session_state.processing_status = "generating_summary"
        
        # Generate summary with progress tracking
        with st.spinner("Generating summary..."):
            # Create progress indicators for summarization
            summary_container = st.container()
            st.session_state.summary_progress_bar = summary_container.progress(0)
            st.session_state.summary_status_text = summary_container.empty()
            st.session_state.summary_status_text.text("Analyzing text and preparing for summarization...")
            
            # Generate summary with callback for progress updates
            summary = generate_summary(
                full_transcription, 
                language=detected_language,
                progress_callback=summary_progress_callback
            )
            st.session_state.summary_progress_bar.progress(1.0)
            st.session_state.summary_status_text.text("Summary generation complete!")
            
            # Display summary
            st.subheader("Summary")
            st.write(summary)
            # Download button for summary
            if summary:
                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )
        
        # Update processing status
        st.session_state.processing_status = "generating_questions"
        
        # Generate questions
        with st.spinner("Generating questions..."):
            question_status = st.empty()
            question_status.info("Generating questions from content...")
            
            try:
                questions = generate_questions(full_transcription)
                st.subheader("Questions")
                
                if not questions:
                    st.info("No questions could be generated from this content.")
                else:
                    for i, question in enumerate(questions, 1):
                        st.write(f"{i}. {question}")
                    # Download button for questions
                    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
                    st.download_button(
                        label="Download Questions",
                        data=questions_text,
                        file_name="questions.txt",
                        mime="text/plain"
                    )
                        
                question_status.success("Questions generated successfully!")
            except Exception as e:
                logger.error(f"Error generating questions: {str(e)}")
                question_status.error("Could not generate questions from this content.")
        
        # Update processing status
        st.session_state.processing_status = "complete"
        
        # Add a combined download button for all content
        if st.session_state.processing_status == "complete":
            # Combine all generated content with section headers
            combined_content = "# TRANSCRIPTION\n\n"
            combined_content += full_transcription
            combined_content += "\n\n# SUMMARY\n\n"
            combined_content += summary
            combined_content += "\n\n# QUESTIONS\n\n"
            combined_content += questions_text
            
            # Create download button for combined content
            st.download_button(
                label="Download All Content",
                data=combined_content,
                file_name="video_content_all.txt",
                mime="text/plain",
                help="Download the full transcription, summary, and questions in a single file"
            )
        
        # Final cleanup
        cleanup_temp_files()
        return True
    
    except Exception as e:
        # Update error status
        st.session_state.processing_status = "error"
        
        # Display detailed error message
        error_container = st.container()
        error_container.error(f"Error processing video: {str(e)}")
        
        # Show more technical details in an expander
        with error_container.expander("Technical Details"):
            st.code(traceback.format_exc())
            
        # Log the error
        logger.error(f"Error processing video: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Cleanup resources even on error
        cleanup_temp_files()
        
        # Memory cleanup
        gc.collect()
        if hasattr(torch, 'cuda'):
            torch.cuda.empty_cache()
            
        return False

# Main app logic
def main():
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Video", "YouTube URL"])
    
    with tab1:
        st.header("Upload a Video File")
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_file is not None:
            try:
                # Create a status message
                upload_status = st.empty()
                upload_status.info("Processing uploaded file...")
                
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    video_path = tmp_file.name
                
                # Add to cleanup list
                st.session_state.cleanup_files.append(video_path)
                
                # Display the video
                st.video(uploaded_file)
                upload_status.success("File uploaded successfully!")
                
                if st.button("Process Video", key="process_uploaded"):
                    process_video(video_path)
            except Exception as e:
                st.error(f"Error handling uploaded file: {str(e)}")
                logger.error(f"Upload error: {str(e)}")
                logger.error(traceback.format_exc())
    
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
                            # Add to cleanup list - will be handled by process_video
                            process_video(video_path, is_youtube=True)
                        except Exception as e:
                            st.error(f"Error downloading YouTube video: {str(e)}")
            else:
                st.error("Invalid YouTube URL. Please enter a valid URL.")

if __name__ == "__main__":
    main()


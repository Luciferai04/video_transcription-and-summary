import os
import tempfile
import yt_dlp
import logging
import time
import uuid
import json
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, session, send_file
from flask_socketio import SocketIO
from flask_cors import CORS
from werkzeug.utils import secure_filename
from modules.audio_processing import extract_audio, filter_silence
from modules.transcription import transcribe_audio, detect_language_confidence
from modules.summarization import summarize_text
from modules.question_generation import generate_questions
from modules.hindi_support import clean_hindi_text
from transformers import pipeline
import threading
import re
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "video_transcription_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('downloads', exist_ok=True)

# Store processing status for each session
processing_status = {}

def format_bytes(bytes_value):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.1f} TB"

def download_with_progress(url, session_id):
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
                        
                        # Format progress message
                        speed_str = f"{format_bytes(speed)}/s" if speed else "..."
                        progress_message = {
                            "type": "download_progress",
                            "percent": percent,
                            "downloaded": format_bytes(downloaded),
                            "total": format_bytes(total),
                            "speed": speed_str
                        }
                        
                        # Update processing status
                        processing_status[session_id]["status"] = "downloading"
                        processing_status[session_id]["progress"] = percent
                        processing_status[session_id]["message"] = f"Downloading... {percent:.1f}%"
                        
                        # Emit progress via WebSocket
                        socketio.emit('progress_update', progress_message, room=session_id)
                except Exception as e:
                    socketio.emit('progress_update', {
                        "type": "download_progress",
                        "error": f"Progress calculation failed: {str(e)}"
                    }, room=session_id)
                    
            elif d['status'] == 'finished':
                socketio.emit('progress_update', {
                    "type": "download_progress",
                    "percent": 100,
                    "message": "Download complete! Processing video..."
                }, room=session_id)
                
                # Update processing status
                processing_status[session_id]["status"] = "processing"
                processing_status[session_id]["progress"] = 100
                processing_status[session_id]["message"] = "Download complete! Processing video..."
                
            elif d['status'] == 'error':
                error_msg = d.get('error', 'Unknown error')
                socketio.emit('progress_update', {
                    "type": "error",
                    "message": f"Download error: {error_msg}"
                }, room=session_id)
                
                # Update processing status
                processing_status[session_id]["status"] = "error"
                processing_status[session_id]["message"] = f"Download error: {error_msg}"

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Prefer mp4 but fall back to other formats
            'outtmpl': 'downloads/%(id)s.%(ext)s',
            'progress_hooks': [progress_hook],
            'quiet': True,
            'no_warnings': True,
            'retries': 10,  # Retry up to 10 times
            'fragment_retries': 10,
            'ignoreerrors': True,  # Continue with next format on error
            'noplaylist': True,  # Download only the video, ignore playlists
            'merge_output_format': 'mp4',  # Try to merge to mp4 when possible
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            logger.info(f"Video downloaded successfully: {video_path}")
            return video_path

    except Exception as e:
        error_msg = str(e)
        if "HTTP Error 429" in error_msg:
            error = "Download rate limited. Please try again later."
        elif "No space left on device" in error_msg:
            error = "Not enough disk space for download."
        elif "Requested format is not available" in error_msg:
            # Try again with a very generic format
            socketio.emit('progress_update', {
                "type": "warning",
                "message": "Requested format is not available. Trying with basic format..."
            }, room=session_id)
            
            try:
                ydl_opts['format'] = 'best/worst'  # Try any available format
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    video_path = ydl.prepare_filename(info)
                    logger.info(f"Video downloaded successfully with fallback format: {video_path}")
                    return video_path
            except Exception as fallback_error:
                error = f"Fallback download failed: {str(fallback_error)}"
        else:
            error = f"Download failed: {error_msg}"
        
        # Update processing status
        processing_status[session_id]["status"] = "error"
        processing_status[session_id]["message"] = error
        
        # Emit error via WebSocket
        socketio.emit('progress_update', {
            "type": "error",
            "message": error
        }, room=session_id)
        
        return None

class TranscriptionResult:
    """Class to store transcription results and related data"""
    def __init__(self, session_id):
        self.session_id = session_id
        self.video_path = None
        self.audio_path = None
        self.filtered_audio_path = None
        self.language = None
        self.is_mixed = False
        self.transcription = None
        self.summary = None
        self.questions = None
        self.translation = None
        self.confidence = 0.0
        self.chat_history = []
        
# Dictionary to store results for each session
transcription_results = {}

# Language options
language_options = {
    "auto": None,
    "en": "en",
    "hi": "hi",
    "hi-en": "hi-en"
}

def get_session_id():
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def process_video(video_path, session_id, language=None):
    """Process video file to extract audio and prepare for transcription"""
    try:
        # Initialize session status and result storage
        processing_status[session_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Starting video processing..."
        }
        
        if session_id not in transcription_results:
            transcription_results[session_id] = TranscriptionResult(session_id)
        
        result = transcription_results[session_id]
        result.video_path = video_path
        
        # Set up audio paths
        video_path_obj = Path(video_path)
        result.audio_path = str(video_path_obj.parent / f"{video_path_obj.stem}_audio.wav")
        result.filtered_audio_path = str(video_path_obj.parent / f"{video_path_obj.stem}_filtered.wav")
        
        # Update status
        processing_status[session_id]["message"] = "Extracting audio..."
        processing_status[session_id]["progress"] = 20
        socketio.emit('progress_update', {
            "type": "process_update",
            "message": "Extracting audio...",
            "progress": 20
        }, room=session_id)
        
        # Extract audio
        extract_audio(video_path, result.audio_path)
        
        # Update status
        processing_status[session_id]["message"] = "Processing audio..."
        processing_status[session_id]["progress"] = 40
        socketio.emit('progress_update', {
            "type": "process_update",
            "message": "Processing audio...",
            "progress": 40
        }, room=session_id)
        
        # Process audio (filter silence)
        filter_silence(result.audio_path, result.filtered_audio_path)
        
        # Update status
        processing_status[session_id]["message"] = "Audio processing complete"
        processing_status[session_id]["progress"] = 60
        socketio.emit('progress_update', {
            "type": "process_update",
            "message": "Audio processing complete",
            "progress": 60
        }, room=session_id)
        
        # Detect language if not provided
        if language is None or language == "auto":
            processing_status[session_id]["message"] = "Detecting language..."
            socketio.emit('progress_update', {
                "type": "process_update",
                "message": "Detecting language...",
                "progress": 65
            }, room=session_id)
            
            result.language, result.is_mixed, result.confidence = detect_language_confidence(result.filtered_audio_path)
            language_display = "Hinglish" if result.is_mixed else result.language.upper()
            
            processing_status[session_id]["message"] = f"Detected language: {language_display}"
            socketio.emit('progress_update', {
                "type": "language_detection",
                "language": language_display,
                "confidence": result.confidence,
                "progress": 70
            }, room=session_id)
        else:
            result.language = language
            result.is_mixed = language == "hi-en"
            if result.is_mixed:
                result.language = "hi"
            
            processing_status[session_id]["message"] = f"Using selected language: {language}"
            socketio.emit('progress_update', {
                "type": "language_selection",
                "language": language,
                "progress": 70
            }, room=session_id)
        
        return True
    except Exception as e:
        error_msg = f"Video processing error: {str(e)}"
        logger.error(error_msg)
        processing_status[session_id]["status"] = "error"
        processing_status[session_id]["message"] = error_msg
        socketio.emit('progress_update', {
            "type": "error",
            "message": error_msg
        }, room=session_id)
        return False

def transcribe_video(session_id):
    """Transcribe processed video audio"""
    try:
        if session_id not in transcription_results:
            raise ValueError("No processed video found for this session")
        
        result = transcription_results[session_id]
        
        # Update status
        processing_status[session_id]["message"] = "Starting transcription..."
        processing_status[session_id]["progress"] = 75
        socketio.emit('progress_update', {
            "type": "process_update",
            "message": "Starting transcription...",
            "progress": 75
        }, room=session_id)
        
        # Custom transcription progress handler for WebSocket updates
        def progress_handler(progress, message):
            processing_status[session_id]["progress"] = 75 + (progress * 20)  # Scale to 75-95%
            processing_status[session_id]["message"] = message
            socketio.emit('progress_update', {
                "type": "transcription_progress",
                "message": message,
                "progress": 75 + (progress * 20)
            }, room=session_id)
        
        # Modify transcribe_audio to accept a progress handler instead of UI container
        # This requires changes to the transcription module to support progress callbacks
        # For now, we'll just call it directly with None for transcription_area
        result.transcription = transcribe_audio(
            result.filtered_audio_path,
            transcription_area=None,
            language=result.language
        )
        
        if result.transcription:
            # Clean Hindi/Hinglish text if needed
            if result.language == "hi" or result.is_mixed:
                result.transcription = clean_hindi_text(result.transcription)
            
            processing_status[session_id]["message"] = "Transcription complete"
            processing_status[session_id]["progress"] = 95
            socketio.emit('progress_update', {
                "type": "transcription_complete",
                "progress": 95
            }, room=session_id)
            
            return True
        else:
            raise ValueError("Transcription failed")
    
    except Exception as e:
        error_msg = f"Transcription error: {str(e)}"
        logger.error(error_msg)
        processing_status[session_id]["status"] = "error"
        processing_status[session_id]["message"] = error_msg
        socketio.emit('progress_update', {
            "type": "error",
            "message": error_msg
        }, room=session_id)
        return False

def generate_summary(session_id):
    """Generate summary for transcribed content"""
    try:
        if session_id not in transcription_results:
            raise ValueError("No transcription found for this session")
        
        result = transcription_results[session_id]
        if not result.transcription:
            raise ValueError("Transcription not available")
        
        # Update status
        processing_status[session_id]["message"] = "Generating summary..."
        socketio.emit('progress_update', {
            "type": "process_update",
            "message": "Generating summary..."
        }, room=session_id)
        
        # Generate summary
        result.summary = summarize_text(result.transcription)
        
        return result.summary
    except Exception as e:
        error_msg = f"Summary generation error: {str(e)}"
        logger.error(error_msg)
        socketio.emit('progress_update', {
            "type": "error",
            "message": error_msg
        }, room=session_id)
        return None

def generate_content_questions(session_id):
    """Generate questions based on transcribed content"""
    try:
        if session_id not in transcription_results:
            raise ValueError("No transcription found for this session")
        
        result = transcription_results[session_id]
        if not result.transcription:
            raise ValueError("Transcription not available")
        
        # Update status
        processing_status[session_id]["message"] = "Generating questions..."
        socketio.emit('progress_update', {
            "type": "process_update",
            "message": "Generating questions..."
        }, room=session_id)
        
        # Generate questions
        result.questions = generate_questions(result.transcription)
        
        return result.questions
    except Exception as e:
        error_msg = f"Question generation error: {str(e)}"
        logger.error(error_msg)
        socketio.emit('progress_update', {
            "type": "error",
            "message": error_msg
        }, room=session_id)
        return None

def translate_hindi_content(session_id):
    """Translate Hindi content to English"""
    try:
        if session_id not in transcription_results:
            raise ValueError("No transcription found for this session")
        
        result = transcription_results[session_id]
        if not result.transcription:
            raise ValueError("Transcription not available")
        
        if result.language != "hi" or result.is_mixed:
            raise ValueError("Translation only available for Hindi content")
        
        # Update status
        processing_status[session_id]["message"] = "Translating content..."
        socketio.emit('progress_update', {
            "type": "process_update",
            "message": "Translating content..."
        }, room=session_id)
        
        # Translate content
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
        result.translation = translator(result.transcription, max_length=1000)[0]["translation_text"]
        
        return result.translation
    except Exception as e:
        error_msg = f"Translation error: {str(e)}"
        logger.error(error_msg)
        socketio.emit('progress_update', {
            "type": "error",
            "message": error_msg
        }, room=session_id)
        return None

def cleanup_session_files(session_id):
    """Clean up temporary files created during processing"""
    try:
        if session_id in transcription_results:
            result = transcription_results[session_id]
            # Remove video and audio files
            for file_path in [result.video_path, result.audio_path, result.filtered_audio_path]:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"Removed temporary file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error removing file {file_path}: {str(e)}")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

# API Endpoints

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video file uploads"""
    try:
        session_id = get_session_id()
        
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check file extension
        allowed_extensions = {'mp4', 'mkv', 'avi', 'mov'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({"error": "File type not supported. Please upload mp4, mkv, avi, or mov files"}), 400
        
        # Initialize processing status
        processing_status[session_id] = {
            "status": "uploading",
            "progress": 0,
            "message": "Uploading video..."
        }
        
        # Save file to uploads directory with secure filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(filepath)
        
        logger.info(f"Video file uploaded: {filepath}")
        
        return jsonify({
            "message": "Video uploaded successfully",
            "session_id": session_id,
            "filepath": filepath
        }), 200
        
    except Exception as e:
        error_msg = f"Upload error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route('/youtube', methods=['POST'])
def process_youtube():
    """Handle YouTube URL processing"""
    try:
        session_id = get_session_id()
        data = request.json
        
        if not data or 'url' not in data:
            return jsonify({"error": "No YouTube URL provided"}), 400
        
        youtube_url = data['url']
        
        # Initialize processing status
        processing_status[session_id] = {
            "status": "initializing",
            "progress": 0,
            "message": "Starting YouTube video download..."
        }
        
        # Start download in a background thread to avoid blocking
        def download_thread():
            video_path = download_with_progress(youtube_url, session_id)
            if video_path:
                # Update processing status
                processing_status[session_id]["status"] = "downloaded"
                processing_status[session_id]["message"] = "Video downloaded successfully"
                socketio.emit('progress_update', {
                    "type": "download_complete",
                    "message": "Video downloaded successfully",
                    "video_path": video_path
                }, room=session_id)
            else:
                # Error status already set in download_with_progress
                pass
        
        thread = threading.Thread(target=download_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "message": "YouTube download started",
            "session_id": session_id
        }), 202
        
    except Exception as e:
        error_msg = f"YouTube processing error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route('/transcribe', methods=['POST'])
def start_transcription():
    """Start video transcription process"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        session_id = data.get('session_id')
        if not session_id:
            session_id = get_session_id()
        
        video_path = data.get('video_path')
        if not video_path or not os.path.exists(video_path):
            return jsonify({"error": "Video path is invalid or not found"}), 400
        
        language = data.get('language', 'auto')
        
        # Start processing in a background thread
        def process_thread():
            # Process video (extract audio, filter silence, detect language)
            if process_video(video_path, session_id, language):
                # Transcribe video
                if transcribe_video(session_id):
                    # Update status
                    processing_status[session_id]["status"] = "completed"
                    processing_status[session_id]["progress"] = 100
                    processing_status[session_id]["message"] = "Transcription complete"
                    socketio.emit('progress_update', {
                        "type": "transcription_complete",
                        "message": "Transcription complete",
                        "progress": 100
                    }, room=session_id)
        
        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "message": "Transcription process started",
            "session_id": session_id
        }), 202
        
    except Exception as e:
        error_msg = f"Transcription start error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route('/status', methods=['GET'])
def check_status():
    """Check processing status"""
    try:
        session_id = request.args.get('session_id')
        if not session_id:
            session_id = get_session_id()
            if session_id not in processing_status:
                return jsonify({"status": "no_session", "message": "No active processing session found"}), 404
        
        if session_id in processing_status:
            return jsonify(processing_status[session_id]), 200
        else:
            return jsonify({"status": "not_found", "message": "Session not found"}), 404
            
    except Exception as e:
        error_msg = f"Status check error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route('/summary', methods=['GET'])
def get_summary():
    """Get summary of transcribed content"""
    try:
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({"error": "No session ID provided"}), 400
        
        if session_id not in transcription_results:
            return jsonify({"error": "No transcription found for this session"}), 404
        
        result = transcription_results[session_id]
        
        # Generate summary if not already generated
        if not result.summary:
            result.summary = generate_summary(session_id)
            
        if result.summary:
            return jsonify({
                "summary": result.summary,
                "transcription": result.transcription
            }), 200
        else:
            return jsonify({"error": "Failed to generate summary"}), 500
            
    except Exception as e:
        error_msg = f"Summary retrieval error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route('/questions', methods=['GET'])
def get_questions():
    """Get questions generated from transcribed content"""
    try:
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({"error": "No session ID provided"}), 400
        
        if session_id not in transcription_results:
            return jsonify({"error": "No transcription found for this session"}), 404
        
        result = transcription_results[session_id]
        
        # Generate questions if not already generated
        if not result.questions:
            result.questions = generate_content_questions(session_id)
            
        if result.questions:
            return jsonify({
                "questions": result.questions,
                "transcription": result.transcription
            }), 200
        else:
            return jsonify({"error": "Failed to generate questions"}), 500
            
    except Exception as e:
        error_msg = f"Questions retrieval error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route('/translate', methods=['POST'])
def translate_content():
    """Translate Hindi content to English"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({"error": "No session ID provided"}), 400
        
        if session_id not in transcription_results:
            return jsonify({"error": "No transcription found for this session"}), 404
        
        result = transcription_results[session_id]
        
        # Verify content is Hindi
        if result.language != "hi":
            return jsonify({"error": "Translation only available for Hindi content"}), 400
        
        # Translate if not already translated
        if not result.translation:
            result.translation = translate_hindi_content(session_id)
            
        if result.translation:
            return jsonify({
                "translation": result.translation,
                "original": result.transcription
            }), 200
        else:
            return jsonify({"error": "Failed to translate content"}), 500
            
    except Exception as e:
        error_msg = f"Translation error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route('/chat', methods=['POST'])
def chat_with_transcription():
    """Chat with AI about the transcribed content"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({"error": "No session ID provided"}), 400
        
        user_message = data.get('message')
        if not user_message or not isinstance(user_message, str):
            return jsonify({"error": "Invalid or missing message"}), 400
            
        if session_id not in transcription_results:
            return jsonify({"error": "No transcription found for this session"}), 404
        
        result = transcription_results[session_id]
        
        if not result.transcription:
            return jsonify({"error": "Transcription not available"}), 400
            
        # Get conversation history if provided
        messages_history = data.get('history', [])
        
        # If no history provided, use stored history
        if not messages_history and result.chat_history:
            messages_history = result.chat_history
            
        # Add the new message to history
        messages_history.append({"role": "user", "content": user_message})
        
        # Generate AI response based on transcription and user message
        response = generate_chat_response(user_message, result.transcription, messages_history)
        
        # Add AI response to history
        messages_history.append({"role": "assistant", "content": response})
        
        # Update stored chat history
        result.chat_history = messages_history
        
        return jsonify({
            "response": response,
            "history": messages_history
        }), 200
            
    except Exception as e:
        error_msg = f"Chat error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

# Initialize sentence transformer model for semantic search (lazy loading)
_sentence_model = None

def get_sentence_model():
    """Lazy load the sentence transformer model"""
    global _sentence_model
    if _sentence_model is None:
        try:
            # Use a smaller, faster model for production
            _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {str(e)}")
            # Return None if model fails to load - we'll handle this in the chat function
    return _sentence_model

def split_into_chunks(text, chunk_size=200, overlap=50):
    """Split text into overlapping chunks for better context preservation"""
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep some overlap for context
            overlap_point = current_chunk.rfind('.', max(0, len(current_chunk) - overlap))
            if overlap_point > 0:
                current_chunk = current_chunk[overlap_point+1:].strip()
            else:
                current_chunk = ""
        
        current_chunk += " " + sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def get_semantic_search_results(query, text_chunks, top_k=3):
    """Find the most semantically similar chunks to the query"""
    model = get_sentence_model()
    if not model:
        # Fall back to keyword matching if model not available
        return keyword_fallback_search(query, text_chunks, top_k)
    
    try:
        # Create embeddings
        query_embedding = model.encode(query, convert_to_tensor=True)
        chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True)
        
        # Calculate cosine similarity
        cos_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
        
        # Get top-k results
        top_results = []
        top_indices = torch.topk(cos_scores, min(top_k, len(text_chunks)))[1]
        
        for idx in top_indices:
            top_results.append({
                'text': text_chunks[idx],
                'score': cos_scores[idx].item()
            })
        
        return top_results
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        # Fall back to keyword matching if semantic search fails
        return keyword_fallback_search(query, text_chunks, top_k)

def keyword_fallback_search(query, text_chunks, top_k=3):
    """Fallback method using keyword matching when semantic search is unavailable"""
    results = []
    query_terms = [term.lower() for term in query.split() if len(term) > 3]
    
    for chunk in text_chunks:
        score = 0
        chunk_lower = chunk.lower()
        
        # Count matches of query terms in chunk
        for term in query_terms:
            score += chunk_lower.count(term)
        
        if score > 0:
            results.append({
                'text': chunk,
                'score': score
            })
    
    # Sort by score and take top_k
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

def extract_question_type(query):
    """Determine the type of question being asked"""
    query_lower = query.lower()
    
    # Define patterns for different question types
    summary_patterns = ["summarize", "summary", "overview", "main points", "gist", "tldr"]
    specific_info_patterns = ["what is", "who is", "when did", "where is", "why did", "how does", "tell me about", "explain"]
    greeting_patterns = ["hello", "hi", "hey", "greetings"]
    farewell_patterns = ["bye", "goodbye", "see you", "farewell"]
    gratitude_patterns = ["thank", "thanks", "appreciate"]
    
    # Check for matches
    if any(pattern in query_lower for pattern in summary_patterns):
        return "summary"
    elif any(pattern in query_lower for pattern in specific_info_patterns):
        return "specific_info"
    elif any(pattern in query_lower for pattern in greeting_patterns):
        return "greeting"
    elif any(pattern in query_lower for pattern in farewell_patterns):
        return "farewell"
    elif any(pattern in query_lower for pattern in gratitude_patterns):
        return "gratitude"
    else:
        return "other"

def get_conversation_context(history, max_context=3):
    """Extract relevant context from conversation history"""
    if not history or len(history) < 2:
        return ""
    
    # Extract last few exchanges, ignoring the current query
    context_messages = []
    for msg in history[:-1][-max_context*2:]:  # Get up to max_context exchanges
        role_prefix = "User: " if msg["role"] == "user" else "Assistant: "
        context_messages.append(f"{role_prefix}{msg['content']}")
    
    if context_messages:
        return "\n".join(context_messages)
    return ""

def generate_chat_response(user_message, transcription, history):
    """Generate a response to the user's message based on the transcription"""
    try:
        # Determine the question type
        question_type = extract_question_type(user_message)
        
        # Get conversation context from history
        conversation_context = get_conversation_context(history)
        
        # Handle different question types
        if question_type == "greeting":
            return "Hello! I can help answer questions about the transcribed content. What would you like to know?"
            
        elif question_type == "farewell":
            return "Goodbye! Feel free to chat again if you have more questions about the transcribed content."
            
        elif question_type == "gratitude":
            return "You're welcome! Let me know if you have any other questions about the transcription."
            
        elif question_type == "summary":
            # If the transcription is short, return it directly
            if len(transcription) < 500:
                return f"Here's a summary of the transcription: {transcription}"
                
            # For longer transcriptions, return initial part or a proper summary if available
            result = None
            for msg in history:
                if msg["role"] == "assistant" and "summary" in msg["content"].lower()[:50]:
                    result = msg["content"]
                    break
            
            if result:
                return f"As I mentioned earlier: {result}"
            else:
                # Extract a reasonable preview
                preview = transcription[:300].strip()
                return f"Here's the beginning of the transcription: \"{preview}...\"\n\nWould you like me to continue with more of the content or focus on a specific part?"
        
        # For specific information or other queries, use semantic search
        # Prepare search query by combining user message with conversation context
        enhanced_query = user_message
        if conversation_context and len(user_message.split()) < 5:
            # For short queries, add context from recent conversation
            enhanced_query = f"{conversation_context}\n{user_message}"
        
        # Split transcription into manageable chunks for semantic processing
        text_chunks = split_into_chunks(transcription)
        if not text_chunks:
            return "I couldn't process the transcription properly. Please try a different question."
        
        # Get semantically relevant chunks
        search_results = get_semantic_search_results(enhanced_query, text_chunks)
        
        # If no results found, try with just the user message without context
        if not search_results and enhanced_query != user_message:
            search_results = get_semantic_search_results(user_message, text_chunks)
        
        # If still no results, provide a general response
        if not search_results:
            return "I couldn't find specific information about that in the transcription. Could you please rephrase your question or ask about another topic?"
        
        # Format the response based on search results
        if len(search_results) == 1 or (len(search_results) > 1 and search_results[0]['score'] > 0.7):
            # One clear match or a very strong top match
            return f"Based on the transcription, I found this information: \"{search_results[0]['text']}\""
        else:
            # Multiple potential matches
            response = "I found several relevant pieces of information in the transcription:\n\n"
            for i, result in enumerate(search_results[:2]):  # Limit to top 2 for readability
                response += f"- \"{result['text']}\"\n\n"
            return response.strip()
        
    except Exception as e:
        logger.error(f"Error generating chat response: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return "Sorry, I encountered an error generating a response. Please try again with a different question."

# WebSocket event handlers

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    session_id = get_session_id()
    logger.info(f"Client connected with session ID: {session_id}")
    
    # Join room based on session ID
    from flask_socketio import join_room
    join_room(session_id)
    
    # Send initial status if available
    if session_id in processing_status:
        socketio.emit('progress_update', {
            "type": "status",
            "data": processing_status[session_id]
        }, room=session_id)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info("Client disconnected")

# Periodic cleanup (runs every hour)
def scheduled_cleanup():
    """Clean up old files and sessions"""
    try:
        # Remove files older than 24 hours
        current_time = time.time()
        max_age = 24 * 60 * 60  # 24 hours in seconds
        
        # Clean up uploads directory
        for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
            for filename in files:
                filepath = os.path.join(root, filename)
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > max_age:
                    try:
                        os.remove(filepath)
                        logger.info(f"Removed old file: {filepath}")
                    except Exception as e:
                        logger.error(f"Error removing old file {filepath}: {str(e)}")
                        
        # Clean up downloads directory
        for root, dirs, files in os.walk('downloads'):
            for filename in files:
                filepath = os.path.join(root, filename)
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > max_age:
                    try:
                        os.remove(filepath)
                        logger.info(f"Removed old file: {filepath}")
                    except Exception as e:
                        logger.error(f"Error removing old file {filepath}: {str(e)}")
        
        # Clean up old sessions
        expired_sessions = []
        for session_id in list(processing_status.keys()):
            # Check if session is old and completed or in error state
            status = processing_status.get(session_id, {}).get("status")
            if status in ["completed", "error"]:
                # Remove from memory
                expired_sessions.append(session_id)
                
        for session_id in expired_sessions:
            cleanup_session_files(session_id)
            if session_id in processing_status:
                del processing_status[session_id]
            if session_id in transcription_results:
                del transcription_results[session_id]
            logger.info(f"Cleaned up session: {session_id}")
            
    except Exception as e:
        logger.error(f"Scheduled cleanup error: {str(e)}")

# Simple health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Video transcription API is running"
    }), 200

# Start cleanup scheduler
def start_cleanup_scheduler():
    """Start periodic cleanup task"""
    import threading
    import time
    
    def cleanup_thread():
        while True:
            try:
                # Run cleanup every hour
                time.sleep(3600)
                logger.info("Running scheduled cleanup")
                scheduled_cleanup()
            except Exception as e:
                logger.error(f"Cleanup scheduler error: {str(e)}")
    
    thread = threading.Thread(target=cleanup_thread)
    thread.daemon = True
    thread.start()
    logger.info("Cleanup scheduler started")

if __name__ == "__main__":
    try:
        # Create necessary directories
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs('downloads', exist_ok=True)
        
        # Start cleanup scheduler
        start_cleanup_scheduler()
        
        # Get host and port from environment variables or use defaults
        import argparse
        parser = argparse.ArgumentParser(description='Video Transcription API Server')
        parser.add_argument('--host', default=os.environ.get('HOST', '0.0.0.0'), help='Host to bind to')
        parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5000)), help='Port to bind to')
        parser.add_argument('--debug', action='store_true', default=os.environ.get('DEBUG', 'False').lower() == 'true', help='Enable debug mode')
        args = parser.parse_args()
        
        # Configure server
        host = args.host
        port = args.port
        debug = args.debug
        
        # Start Flask app with SocketIO
        logger.info(f"Starting Video Transcription API server on {host}:{port} (Debug: {debug})")
        socketio.run(
            app, 
            host=host, 
            port=port, 
            debug=debug, 
            allow_unsafe_werkzeug=debug,  # Only allow unsafe werkzeug in debug mode
            use_reloader=debug            # Only use reloader in debug mode
        )
    except Exception as e:
        logger.error(f"Server startup error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

import os
import sys
import tempfile
import shutil
import logging
import subprocess
import librosa
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import noisereduce as nr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for audio processing
CHUNK_SIZE = 30  # seconds
MIN_SILENCE_LEN = 0.5  # seconds
SILENCE_THRESH = -40  # dB
KEEP_SILENCE = 0.3  # seconds
NOISE_REDUCE_PROP = 0.5  # noise reduction strength (0-1)

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        return True
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install FFmpeg first.")
        return False

def extract_audio(video_path, audio_path):
    """Extract audio from video file"""
    try:
        if not check_ffmpeg():
            raise RuntimeError("FFmpeg is required but not found")
        
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # Use PCM format
            '-ar', '16000',  # Set sample rate
            '-ac', '1',  # Convert to mono
            '-y',  # Overwrite output file
            audio_path
        ]
        
        process = subprocess.run(command, capture_output=True, text=True)
        
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {process.stderr}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return False

def filter_silence(input_path, output_path):
    """Remove silence from audio file with noise reduction"""
    try:
        # Load the audio file
        y, sr = librosa.load(input_path, sr=None)
        
        # Detect non-silent intervals
        intervals = librosa.effects.split(
            y,
            top_db=abs(SILENCE_THRESH),
            frame_length=int(MIN_SILENCE_LEN * sr),
            hop_length=int(KEEP_SILENCE * sr)
        )
        
        # Concatenate non-silent parts
        y_filtered = np.concatenate([y[start:end] for start, end in intervals])
        
        # Apply noise reduction
        y_reduced = nr.reduce_noise(
            y=y_filtered,
            sr=sr,
            prop_decrease=NOISE_REDUCE_PROP,
            stationary=True
        )
        
        # Save the processed audio
        sf.write(output_path, y_reduced, sr)
        
        logger.info(f"Successfully filtered audio to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error filtering silence: {str(e)}")
        return False

def validate_audio_file(file_path):
    """Validate audio file format and quality"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return False
        
        y, sr = librosa.load(file_path, sr=None)
        
        # Check sample rate
        if sr < 8000:
            logger.warning(f"Low sample rate detected: {sr} Hz")
        
        # Check duration
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 1.0:
            logger.warning(f"Very short audio detected: {duration:.2f} seconds")
        
        # Check amplitude
        rms = np.sqrt(np.mean(y**2))
        if rms < 0.01:
            logger.warning("Low amplitude detected")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating audio file: {str(e)}")
        return False

def get_audio_stats(audio_path):
    """Get audio file statistics"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.sqrt(np.mean(y**2))
        peak = np.max(np.abs(y))
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'rms_level': float(rms),
            'peak_level': float(peak)
        }
        
    except Exception as e:
        logger.error(f"Error getting audio stats: {str(e)}")
        return None

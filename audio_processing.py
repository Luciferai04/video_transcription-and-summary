import os
import subprocess
import logging
import numpy as np
import librosa
import torch
import torchaudio
import io
import shutil
from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy import signal
import importlib.util
import tempfile
import noisereduce as nr

logger = logging.getLogger(__name__)

def _check_import(module_name, test_import=None):
    """Check if a module and optional sub-import are available."""
    try:
        module = importlib.import_module(module_name)
        if test_import:
            getattr(module, test_import)
        logger.info("Successfully checked import: %s", module_name)
        return True
    except (ImportError, AttributeError) as e:
        logger.warning("Failed to import %s (test: %s): %s", module_name, test_import, str(e))
        return False

# Check for moviepy, ffmpeg, and speechbrain
USE_MOVIEPY = _check_import("moviepy.editor", "VideoFileClip")
USE_FFMPEG_PYTHON = _check_import("ffmpeg")
USE_SPEECHBRAIN = False  # Default to False

# Enhanced SpeechBrain check
if _check_import("speechbrain.pretrained", "SpectralMaskEnhancement"):
    try:
        from speechbrain.pretrained import SpectralMaskEnhancement
        # Test if the model can be loaded and has a forward method
        enhancer = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir="pretrained_models/metricgan-plus-voicebank",
            run_opts={"device": "cpu"}  # Force CPU usage for SpeechBrain
        )
        # Check for forward method
        if not hasattr(enhancer, 'forward'):
            raise AttributeError("SpectralMaskEnhancement is missing the 'forward' method")
        USE_SPEECHBRAIN = True
        logger.info("SpeechBrain is available and functional")
    except Exception as e:
        logger.warning("SpeechBrain setup failed: %s. Falling back to noisereduce.", str(e))
        USE_SPEECHBRAIN = False

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
if not FFMPEG_AVAILABLE:
    logger.warning("ffmpeg binary not found. Install with 'brew install ffmpeg' to ensure audio extraction fallback.")

logger.info(f"Video processing config — MoviePy: {USE_MOVIEPY}, ffmpeg-python: {USE_FFMPEG_PYTHON}, ffmpeg binary: {FFMPEG_AVAILABLE}, SpeechBrain: {USE_SPEECHBRAIN}")

if USE_MOVIEPY:
    try:
        from moviepy.editor import VideoFileClip
        logger.info("Imported VideoFileClip from moviepy.editor")
    except ImportError as e:
        logger.error("Failed to import VideoFileClip despite initial check: %s", str(e))
        USE_MOVIEPY = False

def convert_to_wav(input_path, output_path):
    """Convert audio file to WAV format using ffmpeg."""
    try:
        if not FFMPEG_AVAILABLE:
            raise EnvironmentError("ffmpeg not found. Install with 'brew install ffmpeg'")
        cmd = ["ffmpeg", "-i", input_path, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", output_path]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info("Converted %s to WAV: %s", input_path, output_path)
    except Exception as e:
        logger.error("Failed to convert to WAV: %s", str(e))
        raise

def load_audio(audio_path, sr=16000):
    """Load and resample audio to specified sample rate using torchaudio."""
    try:
        if not audio_path.endswith('.wav'):
            temp_wav = "temp_input.wav"
            convert_to_wav(audio_path, temp_wav)
            audio_path = temp_wav

        # Use torchaudio with soundfile backend
        waveform, sample_rate = torchaudio.load(audio_path, backend="soundfile")
        # Convert to numpy and ensure mono
        audio = waveform.numpy()
        if audio.shape[0] > 1:
            audio = audio.mean(axis=0)  # Convert to mono
        else:
            audio = audio[0]

        # Resample if needed
        if sample_rate != sr:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=sr)
            sample_rate = sr

        logger.info("Loaded audio: %s, sample rate: %d Hz, shape: %s", audio_path, sample_rate, audio.shape)
        if audio_path == temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)
        return audio, sample_rate
    except Exception as e:
        logger.error("Failed to load audio: %s", str(e))
        raise

def apply_bandpass_filter(audio, sr=16000, lowcut=100, highcut=8000):
    """Apply bandpass filter to focus on speech frequencies."""
    try:
        if lowcut >= highcut or lowcut <= 0 or highcut >= sr / 2:
            logger.warning("Invalid bandpass filter parameters: lowcut=%d, highcut=%d, sr=%d", lowcut, highcut, sr)
            return audio
        nyquist = 0.5 * sr
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, audio)
        filtered = librosa.effects.preemphasis(filtered)
        filtered = librosa.effects.deemphasis(filtered)
        logger.info("Bandpass filter applied: lowcut=%d Hz, highcut=%d Hz", lowcut, highcut)
        return filtered
    except Exception as e:
        logger.error("Bandpass filter error: %s", str(e))
        return audio

def reduce_noise_noisereduce(audio, sr=16000):
    """Apply noise reduction using the noisereduce library."""
    try:
        noise_clip = audio[:int(sr * 0.5)]
        reduced = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_clip, prop_decrease=0.8)
        logger.info("Noise reduction applied using noisereduce")
        return reduced
    except Exception as e:
        logger.error("Noise reduction with noisereduce failed: %s", str(e))
        return audio

def enhance_audio(audio, sr=16000):
    """Apply advanced noise reduction using SpeechBrain MetricGAN+ if available, else use noisereduce."""
    if USE_SPEECHBRAIN:
        try:
            from speechbrain.pretrained import SpectralMaskEnhancement
            enhancer = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                savedir="pretrained_models/metricgan-plus-voicebank",
                run_opts={"device": "cpu"}  # Force CPU usage for SpeechBrain
            )
            # Explicitly use CPU tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32, device="cpu").unsqueeze(0)
            logger.debug("Audio tensor shape: %s, sample rate: %d", audio_tensor.shape, sr)
            if sr != 16000:
                resampled = librosa.resample(audio_tensor.cpu().numpy()[0], orig_sr=sr, target_sr=16000)
                audio_tensor = torch.tensor(resampled, dtype=torch.float32, device="cpu").unsqueeze(0)
                logger.info("Resampled audio to 16kHz for MetricGAN+")
            # Ensure enhancer is using CPU and handle the output properly
            with torch.no_grad():
                try:
                    enhanced = enhancer.enhance_batch(audio_tensor)
                except AttributeError:
                    # Try alternative method if enhance_batch is not available
                    enhanced = enhancer(audio_tensor)
                
                enhanced = enhanced.squeeze().cpu().numpy()
            logger.info("Audio enhanced with MetricGAN+")
            return enhanced
        except Exception as e:
            logger.error("Enhancement error with SpeechBrain: %s", str(e))
            logger.info("Falling back to noisereduce for noise reduction")
            return reduce_noise_noisereduce(audio, sr)
    else:
        logger.info("SpeechBrain not available; using noisereduce for noise reduction.")
        return reduce_noise_noisereduce(audio, sr)

def extract_audio_with_moviepy(video_path, audio_path):
    """Extract audio using MoviePy."""
    try:
        logger.info("Using MoviePy for extraction")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        video.close()
        logger.info("MoviePy extraction successful")
        return True
    except Exception as e:
        logger.error("MoviePy extraction failed: %s", str(e))
        return False

def extract_audio_with_ffmpeg(video_path, audio_path):
    """Extract audio using ffmpeg."""
    logger.info("Using ffmpeg for extraction")
    try:
        if USE_FFMPEG_PYTHON:
            import ffmpeg
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream.audio, audio_path, acodec="pcm_s16le", ar="16000", ac=1)
            ffmpeg.run(stream, quiet=True, overwrite_output=True)
        else:
            if not FFMPEG_AVAILABLE:
                raise EnvironmentError("ffmpeg not found. Install with 'brew install ffmpeg'")
            cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", audio_path]
            subprocess.run(cmd, check=True, capture_output=True)
        logger.info("FFmpeg extraction successful")
        return True
    except Exception as e:
        logger.error("FFmpeg extraction failed: %s", str(e))
        return False

def extract_audio(video_path, audio_path):
    """Extract audio from video using MoviePy or ffmpeg."""
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv']:
            raise ValueError(f"Unsupported video format: {ext}")
        logger.info("Attempting audio extraction from: %s", video_path)
        success = USE_MOVIEPY and extract_audio_with_moviepy(video_path, audio_path)
        if not success:
            logger.info("Falling back to ffmpeg extraction")
            success = extract_audio_with_ffmpeg(video_path, audio_path)
        if not success or not os.path.exists(audio_path):
            raise RuntimeError("Audio extraction failed")
        if os.path.getsize(audio_path) < 1000:
            raise RuntimeError("Extracted audio is too small to be valid")
        logger.info("Audio extracted successfully: %s", audio_path)
    except Exception as e:
        logger.error("Extraction failed: %s", str(e))
        raise

def filter_silence(audio_path, output_path, silence_thresh=-35, min_silence_len=300):
    """Remove silent segments and enhance audio."""
    try:
        audio, sr = load_audio(audio_path)
        audio = apply_bandpass_filter(audio, sr)
        audio = enhance_audio(audio, sr)
        audio_seg = AudioSegment(
            (audio * 32767).astype(np.int16).tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        chunks = split_on_silence(
            audio_seg,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=150
        )
        if not chunks:
            logger.warning("No speech chunks detected; using full audio")
            chunks = [audio_seg]
        filtered = AudioSegment.empty()
        for c in chunks:
            filtered += c
        filtered.export(output_path, format="wav")
        logger.info("Filtered and saved audio: %s", output_path)
    except Exception as e:
        logger.error("Silence filtering failed: %s", str(e))
        raise
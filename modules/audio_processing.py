import os
import subprocess
import logging
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy import signal
import noisereduce as nr
import whisper
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE = 16000  # Whisper's preferred sample rate

def estimate_noise_profile(audio, sr, frame_duration=0.5):
    """Estimate noise profile from the quietest sections of audio."""
    # Convert frame duration to samples
    frame_length = int(frame_duration * sr)
    
    # Calculate energy for each frame
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=frame_length)[0]
    
    # Find the indices of the 10% quietest frames
    num_quiet_frames = max(1, len(energy) // 10)
    quiet_indices = np.argsort(energy)[:num_quiet_frames]
    
    # Extract and concatenate quiet frames
    noise_profile = []
    for idx in quiet_indices:
        start = idx * frame_length
        end = start + frame_length
        if end <= len(audio):
            noise_profile.append(audio[start:end])
    
    return np.concatenate(noise_profile) if noise_profile else audio[:frame_length]

def apply_adaptive_noise_reduction(audio, sr):
    """Apply adaptive noise reduction based on signal characteristics."""
    try:
        # Estimate noise profile
        noise_profile = estimate_noise_profile(audio, sr)
        
        # Apply noise reduction with adaptive parameters
        # Calculate signal-to-noise ratio
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise_profile ** 2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        
        # Adjust noise reduction parameters based on SNR
        if snr < 10:  # High noise
            prop_decrease = 0.85
            stationary = True
        elif snr < 20:  # Moderate noise
            prop_decrease = 0.70
            stationary = True
        else:  # Low noise
            prop_decrease = 0.50
            stationary = False
        
        # Apply noise reduction
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            y_noise=noise_profile,
            prop_decrease=prop_decrease,
            stationary=stationary,
            n_jobs=-1
        )
        
        return reduced
    except Exception as e:
        logger.error(f"Adaptive noise reduction failed: {str(e)}")
        return audio

def apply_spectral_gating(audio, sr):
    """Apply spectral gating for additional noise reduction."""
    try:
        # Parameters for spectral gating
        n_fft = 2048
        win_length = 2048
        hop_length = 512
        
        # Compute STFT
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        
        # Compute magnitude spectrogram
        mag = np.abs(D)
        
        # Estimate noise floor
        noise_floor = np.mean(np.sort(mag, axis=1)[:, :int(mag.shape[1]/10)], axis=1)
        
        # Create spectral gate
        threshold = 2.0  # Adjust based on noise level
        gate = (mag.T > threshold * noise_floor).T
        
        # Apply gate
        D_gated = D * gate
        
        # Inverse STFT
        audio_gated = librosa.istft(D_gated, hop_length=hop_length, win_length=win_length)
        
        return audio_gated
    except Exception as e:
        logger.error(f"Spectral gating failed: {str(e)}")
        return audio

def enhance_audio(audio, sr):
    """Apply comprehensive audio enhancement optimized for speech recognition."""
    try:
        # 1. Initial normalization
        audio = librosa.util.normalize(audio)
        
        # 2. Apply adaptive noise reduction
        audio = apply_adaptive_noise_reduction(audio, sr)
        
        # 3. Apply spectral gating
        audio = apply_spectral_gating(audio, sr)
        
        # 4. Apply bandpass filter to focus on speech frequencies
        nyquist = sr / 2
        low = 80 / nyquist
        high = 7500 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        audio = signal.filtfilt(b, a, audio)
        
        # 5. Apply dynamic range compression
        percentile = np.percentile(np.abs(audio), 95)
        audio = np.clip(audio, -percentile, percentile)
        audio = librosa.util.normalize(audio)
        
        # 6. Final noise reduction pass with gentle settings
        audio = nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=0.5,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_jobs=-1
        )
        
        return audio
    except Exception as e:
        logger.error(f"Audio enhancement failed: {str(e)}")
        return librosa.util.normalize(audio)  # Return normalized original if enhancement fails

def convert_audio_format(input_path, output_path, sr=AUDIO_SAMPLE_RATE):
    """Convert audio to WAV format using ffmpeg with noise reduction."""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # First conversion with initial noise reduction
        temp_path = output_path + ".temp.wav"
        cmd = [
            "ffmpeg", "-i", input_path,
            "-af", "highpass=f=80,lowpass=f=7500,areverse,silenceremove=start_periods=1:start_threshold=-50dB,areverse",  # Basic noise filtering
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-ac", "1",
            "-y",
            temp_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Load the temporary file for additional processing
        audio, sr = sf.read(temp_path)
        
        # Apply enhanced noise reduction
        enhanced = enhance_audio(audio, sr)
        
        # Save the final enhanced audio
        sf.write(output_path, enhanced, sr)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        logger.info(f"Converted and enhanced audio: {input_path} -> {output_path}")
        return True
    except Exception as e:
        logger.error(f"Audio conversion failed: {str(e)}")
        return False

def filter_silence(audio_path, output_path, min_silence_len=500, silence_thresh=-40):
    """Process audio with advanced filtering and noise reduction."""
    try:
        # Load audio
        audio, sr = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if sr != AUDIO_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_SAMPLE_RATE)
            sr = AUDIO_SAMPLE_RATE
        
        # Apply comprehensive enhancement
        enhanced = enhance_audio(audio, sr)
        
        # Save to temporary file
        temp_path = output_path + ".temp.wav"
        sf.write(temp_path, enhanced, sr)
        
        # Load with pydub for silence removal
        audio_segment = AudioSegment.from_wav(temp_path)
        
        # Split on silence with more aggressive settings for better segmentation
        chunks = split_on_silence(
            audio_segment,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=200,  # Keep 200ms of silence between chunks
            seek_step=1  # More precise silence detection
        )
        
        # Process and combine chunks
        if not chunks:
            logger.warning("No non-silent chunks found, using processed audio")
            processed_audio = audio_segment
        else:
            processed_audio = AudioSegment.empty()
            for chunk in chunks:
                # Add small fade in/out to prevent clicks
                chunk = chunk.fade_in(10).fade_out(10)
                processed_audio += chunk
        
        # Normalize final audio
        processed_audio = processed_audio.normalize()
        
        # Export processed audio
        processed_audio.export(output_path, format="wav")
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        logger.info(f"Processed and saved enhanced audio: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        raise

def extract_audio(video_path, audio_path):
    """Extract audio from video with optimal settings."""
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(audio_path)), exist_ok=True)
        
        # Extract audio with initial noise reduction
        temp_path = audio_path + ".temp.wav"
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn",  # No video
            "-af", "highpass=f=80,lowpass=f=7500",  # Basic filtering
            "-acodec", "pcm_s16le",
            "-ar", str(AUDIO_SAMPLE_RATE),
            "-ac", "1",
            "-y",
            temp_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Load the temporary file
        audio, sr = sf.read(temp_path)
        
        # Apply enhanced noise reduction
        enhanced = enhance_audio(audio, sr)
        
        # Save the final enhanced audio
        sf.write(audio_path, enhanced, sr)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        logger.info(f"Audio extracted and enhanced: {audio_path}")
        return True
    except Exception as e:
        logger.error(f"Audio extraction failed: {str(e)}")
        raise

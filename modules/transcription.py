import whisper
import streamlit as st
import logging
import os
import torch
import re
import numpy as np
import soundfile as sf
from pathlib import Path
import time
import gc
from functools import lru_cache

logger = logging.getLogger(__name__)

HINDI_PROMPT = """
This is a transcription of Hindi content. Please transcribe in romanized Hindi (Hinglish) instead of Devanagari script.
Examples:
- "मैं आज बहुत खुश हूँ" should be written as "Main aaj bahut khush hoon"
- "यह एक अच्छा वीडियो है" should be written as "Yeh ek accha video hai"
- "आपका स्वागत है" should be written as "Aapka swagat hai"
IMPORTANT: Use ONLY Roman script (English alphabets) for ALL Hindi words. NEVER use Devanagari script in the output.
Ensure natural spelling in romanized form and maintain the meaning of the original content.
"""

HINGLISH_PROMPT = """
This is a transcription of mixed Hindi-English content that may include:
- Technical discussions
- Educational lectures
- Professional conversations
Please maintain natural code-switching between Hindi and English.
Output all Hindi words in romanized form (using Roman script) rather than Devanagari.
Examples:
- "मैं computer use करता हूँ" should be written as "Main computer use karta hoon"
- "यह important topic है" should be written as "Yeh important topic hai"
- "क्या आप marketing strategy समझा सकते हैं" should be written as "Kya aap marketing strategy samjha sakte hain"
IMPORTANT: Use ONLY Roman script (English alphabets) for ALL Hindi words. NEVER use Devanagari script in the output.
Ensure the transcription maintains the mixed language nature with English technical terms preserved.
"""

@lru_cache(maxsize=1)
def get_whisper_model(model_name="large", device="cpu"):
    """Cache the model to prevent repeated loading."""
    logger.info(f"Loading Whisper {model_name} model on {device}")
    try:
        model = whisper.load_model(model_name)
        model = model.to(device)
        return model
    except Exception as e:
        logger.error(f"Error loading Whisper model: {str(e)}")
        raise

def load_audio_chunk(file_path, chunk_size=480000, offset=0):
    """Load a chunk of audio with proper memory handling."""
    try:
        # Try to load using whisper's native function first
        try:
            audio = whisper.load_audio(file_path)
        except Exception as whisper_error:
            logger.warning(f"Whisper failed to load audio: {str(whisper_error)}")
            # Fallback to alternative loading method if whisper fails
            import librosa
            logger.info("Trying to load audio with librosa...")
            audio, _ = librosa.load(file_path, sr=16000, mono=True)
            
        if offset >= len(audio):
            return None
        end = min(offset + chunk_size, len(audio))
        return audio[offset:end]
    except Exception as e:
        logger.error(f"Error loading audio chunk: {str(e)}")
        return None

def detect_language_confidence(audio_path):
    """Detect language with confidence score, with improved English detection."""
    try:
        # Load just a 30-second chunk for language detection
        audio = load_audio_chunk(audio_path, chunk_size=480000)  # 30 seconds at 16kHz
        if audio is None:
            logger.warning("Could not load audio for language detection, using English as fallback")
            return "en", False, 0.8

        # Ensure audio is the right shape and format
        audio = whisper.pad_or_trim(audio)
        
        # Load model on CPU
        model = get_whisper_model("large", device="cpu")
        
        try:
            # Get mel spectrogram with enhanced error handling
            try:
                # Try using whisper's built-in mel spectrogram function
                mel = whisper.log_mel_spectrogram(audio).to("cpu")
                
                # Sometimes the mel spectrogram has the wrong shape for detection
                # Check dimensions and reshape if needed
                if mel.dim() == 2:  # If it's missing the batch dimension
                    mel = mel.unsqueeze(0)  # Add batch dimension
                    logger.info("Reshaped mel spectrogram to add batch dimension")
                    
                # Explicitly handle other potential shape issues
                expected_channels = 128  # Required by the model architecture
                if mel.shape[1] != expected_channels:
                    logger.warning(f"Mel spectrogram has {mel.shape[1]} channels instead of expected {expected_channels}")
                    logger.info("Using English as fallback language due to channel mismatch")
                    return "en", False, 0.75
                    
            except Exception as mel_error:
                logger.warning(f"Error generating mel spectrogram: {str(mel_error)}")
                logger.info("Using English as fallback language")
                return "en", False, 0.75
            
            # Detect language with timeout protection
            import threading
            import time
            
            result = {"success": False, "langs": None}
            
            def detect_lang():
                try:
                    _, probs = model.detect_language(mel)
                    result["success"] = True
                    result["langs"] = probs
                except Exception as detect_error:
                    logger.error(f"Language detection internal error: {str(detect_error)}")
            
            # Run detection in thread with timeout
            detection_thread = threading.Thread(target=detect_lang)
            detection_thread.start()
            detection_thread.join(timeout=10)  # 10 second timeout
            
            if not result["success"]:
                logger.warning("Language detection timed out or failed, using English as fallback")
                return "en", False, 0.7
            
            # Get top 2 languages and their probabilities
            probs = result["langs"]
            top_langs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:2]
            
            # Enhanced English detection
            # If English is in top 2 with decent confidence, prioritize it for better results
            if "en" in [lang[0] for lang in top_langs]:
                en_lang = next((lang for lang in top_langs if lang[0] == "en"), None)
                if en_lang and en_lang[1] > 0.35:  # If English confidence is reasonable
                    logger.info(f"Prioritizing English detection with confidence: {en_lang[1]:.2f}")
                    return "en", False, en_lang[1]
            
            # Check for Hindi/English mixing
            if top_langs[0][0] in ["hi", "en"] and top_langs[1][0] in ["hi", "en"]:
                if abs(top_langs[0][1] - top_langs[1][1]) < 0.3:
                    logger.info("Detected potential code-mixing between Hindi and English")
                    return "hi", True, top_langs[0][1]
            
            logger.info(f"Detected language: {top_langs[0][0]} with confidence: {top_langs[0][1]:.2f}")
            return top_langs[0][0], False, top_langs[0][1]
        except RuntimeError as re:
            # Handle the dimension mismatch error specifically
            if "channels" in str(re) or "dimensions" in str(re).lower():
                logger.warning(f"Dimension mismatch in Whisper model: {str(re)}")
                logger.info("Using English as fallback language due to model mismatch")
                return "en", False, 0.8
            else:
                raise
    except Exception as e:
        logger.error(f"Language detection error: {str(e)}")
        logger.info("Falling back to English language detection")
        return "en", False, 0.8

def process_audio_chunk(chunk, model, language, is_mixed, prompt=""):
    """Process a single audio chunk."""
    try:
        # Ensure chunk is properly padded/trimmed
        chunk = whisper.pad_or_trim(chunk)
        
        # Enhanced parameters for English content
        transcription_options = {
            "language": language,
            "initial_prompt": prompt,
            "condition_on_previous_text": True,
            "fp16": False
        }
        
        # Add English-specific optimizations
        if language == "en":
            transcription_options.update({
                # Better parameters for English speech recognition
                "temperature": 0,        # Use greedy decoding for more predictable results
                "beam_size": 5,          # Use beam search for better quality
                "best_of": 5,            # Return best result from beam search
                "patience": 1.0,         # Higher patience for more diverse beams
                "suppress_tokens": [-1]  # Suppress special tokens
            })
        
        # Transcribe chunk with optimized parameters
        result = model.transcribe(chunk, **transcription_options)
        
        return result["text"].strip()
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        return ""

def preprocess_audio_for_english(audio_path):
    """Apply English-specific audio preprocessing for better transcription results."""
    try:
        # Import audio processing libraries if available
        import numpy as np
        from scipy import signal
        import soundfile as sf
        import tempfile
        import os
        import subprocess
        
        # Check if file exists and is readable
        if not os.path.isfile(audio_path):
            logger.error(f"File not found: {audio_path}")
            return audio_path
            
        # Try to use ffmpeg to convert the file to a WAV first to ensure compatibility
        temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav_path = temp_wav_file.name
        temp_wav_file.close()
        
        try:
            # Use ffmpeg to convert to WAV format
            command = ["ffmpeg", "-i", audio_path, "-ar", "16000", "-ac", "1", "-y", temp_wav_path]
            subprocess.run(command, check=True, capture_output=True)
            logger.info(f"Converted audio file to WAV format at {temp_wav_path}")
            
            # Now try to load with soundfile
            audio_data, sample_rate = sf.read(temp_wav_path)
        except Exception as ffmpeg_error:
            logger.warning(f"Error converting file with ffmpeg: {str(ffmpeg_error)}")
            
            # Try loading with librosa as fallback
            try:
                import librosa
                audio_data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
                logger.info("Successfully loaded audio with librosa")
            except Exception as librosa_error:
                logger.error(f"Error loading audio with librosa: {str(librosa_error)}")
                return audio_path
        
        # If stereo, convert to mono
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize audio
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
        
        # Apply a simple highpass filter to reduce background noise
        sos = signal.butter(10, 80, 'hp', fs=sample_rate, output='sos')
        filtered_audio = signal.sosfilt(sos, audio_data)
        
        # Save processed audio to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, filtered_audio, sample_rate)
        logger.info(f"Applied English audio preprocessing to {audio_path}")
        
        return temp_file.name
    except Exception as e:
        logger.error(f"Error in English audio preprocessing: {str(e)}")
        logger.info("Using original audio file")
        return audio_path


def transcribe_audio(audio_path, transcription_area=None, language=None):
    """Transcribe audio with improved memory handling and error recovery."""
    start_time = time.time()
    
    # Initialize variables that need to be accessed throughout the function
    is_mixed = False
    confidence = 0.0
    
    try:
        # Detect language if not provided
        if language is None:
            language, is_mixed, confidence = detect_language_confidence(audio_path)
            logger.info(f"Detected language: {language} (mixed: {is_mixed}, confidence: {confidence:.2f})")
        else:
            # Convert string language name to code if needed
            if language.lower() == "english":
                language = "en"
                confidence = 1.0  # Set confidence to 1.0 when language is explicitly provided
            elif language.lower() == "hindi":
                language = "hi"
                confidence = 1.0  # Set confidence to 1.0 when language is explicitly provided
                
            is_mixed = language == "hi-en"
            if is_mixed:
                language = "hi"
        
        # Apply English-specific preprocessing if needed
        processed_audio_path = audio_path
        if language == "en" and not is_mixed:
            if transcription_area:
                preprocess_status = transcription_area.empty()
                preprocess_status.info("Preprocessing audio for enhanced English transcription...")
            processed_audio_path = preprocess_audio_for_english(audio_path)
        
        # Update UI with language detection
        if transcription_area:
            status = transcription_area.empty()
            lang_display = "English" if language == "en" else "Hinglish" if is_mixed or language == "hi" else language.upper()
            status.info(f"Detected {lang_display} content (Confidence: {confidence:.2f})")
        
        # Initialize progress tracking
        if transcription_area:
            progress_bar = transcription_area.progress(0)
            progress_text = transcription_area.empty()
            output_area = transcription_area.empty()
        
        # Load model
        model = get_whisper_model("large", device="cpu")
        
        # Get appropriate prompt
        # Use HINDI_PROMPT for Hindi to get romanized output, and HINGLISH_PROMPT for mixed content
        # Both prompts instruct the model to use romanized text (Hinglish) instead of Devanagari
        if is_mixed:
            prompt = HINGLISH_PROMPT
            logger.info("Using Hinglish prompt for mixed Hindi-English content")
        elif language == "hi":
            prompt = HINDI_PROMPT
            logger.info("Using Hindi prompt with romanized output instructions")
        else:
            prompt = ""
            logger.info(f"No special prompt for language: {language}")
        
        # Process audio in chunks
        chunk_size = 480000  # 30 seconds at 16kHz
        offset = 0
        transcription = []
        chunk_count = 0
        
        while True:
            # Load chunk
            chunk = load_audio_chunk(audio_path, chunk_size, offset)
            if chunk is None or len(chunk) == 0:
                break
            
            # Process chunk
            chunk_text = process_audio_chunk(chunk, model, language, is_mixed, prompt)
            if chunk_text:
                # Ensure no Devanagari characters in output if language is Hindi
                if language == "hi" and re.search(r"[ऀ-ॿ]", chunk_text):
                    logger.warning("Detected Devanagari in output despite romanization prompt. Attempting cleanup.")
                    # Try to replace common Devanagari characters with romanized equivalents
                    # This is a fallback and not a comprehensive solution
                    from modules.hindi_support import clean_hindi_text
                    chunk_text = clean_hindi_text(chunk_text)
                transcription.append(chunk_text)
            
            # Update progress
            if transcription_area:
                progress = min(1.0, (chunk_count + 1) * 0.1)  # Approximate progress
                progress_bar.progress(progress)
                progress_text.text(f"Processing chunk {chunk_count + 1}")
                output_area.text_area(
                    "Current transcription",
                    " ".join(transcription),
                    height=200
                )
            
            # Move to next chunk
            offset += chunk_size
            chunk_count += 1
            
            # Clear memory
            gc.collect()
            if hasattr(torch, 'cuda'):
                torch.cuda.empty_cache()
        
        # Combine all transcriptions
        final_text = " ".join(transcription)
        
        # Ensure final text is properly cleaned and has no Devanagari if the language is Hindi
        if language == "hi":
            from modules.hindi_support import clean_hindi_text
            final_text = clean_hindi_text(final_text)
        
        # Calculate statistics
        total_time = time.time() - start_time
        duration = offset / 16000  # Convert samples to seconds
        speed_factor = duration / total_time
        logger.info(f"Transcription completed in {total_time:.2f}s ({speed_factor:.2f}x real-time)")
        
        # Update UI with final results
        if transcription_area:
            progress_bar.progress(1.0)
            progress_text.text("Transcription complete!")
            output_area.text_area(
                "Final Transcription",
                final_text,
                height=300
            )
        
        return final_text
    
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        if transcription_area:
            transcription_area.error(f"Error: {str(e)}")
        return None

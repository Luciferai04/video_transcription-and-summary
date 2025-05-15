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
यह एक हिंदी या हिंग्लिश वीडियो की ट्रांसक्रिप्शन है। इसमें शामिल हो सकते हैं:
- शैक्षिक सामग्री और व्याख्यान
- व्यावसायिक चर्चा और प्रस्तुतियां
- तकनीकी विषयों पर चर्चा
"""

HINGLISH_PROMPT = """
This is a transcription of mixed Hindi-English content that may include:
- Technical discussions
- Educational lectures
- Professional conversations
Please maintain natural code-switching between Hindi and English.
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
        audio = whisper.load_audio(file_path)
        if offset >= len(audio):
            return None
        end = min(offset + chunk_size, len(audio))
        return audio[offset:end]
    except Exception as e:
        logger.error(f"Error loading audio chunk: {str(e)}")
        return None

def detect_language_confidence(audio_path):
    """Detect language with confidence score."""
    try:
        # Load just a 30-second chunk for language detection
        audio = load_audio_chunk(audio_path, chunk_size=480000)  # 30 seconds at 16kHz
        if audio is None:
            raise ValueError("Could not load audio for language detection")

        # Ensure audio is the right shape
        audio = whisper.pad_or_trim(audio)
        
        # Load model on CPU
        model = get_whisper_model("large", device="cpu")
        
        # Get mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to("cpu")
        
        # Detect language
        _, probs = model.detect_language(mel)
        
        # Get top 2 languages and their probabilities
        top_langs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:2]
        
        # Check for Hindi/English mixing
        if top_langs[0][0] in ["hi", "en"] and top_langs[1][0] in ["hi", "en"]:
            if abs(top_langs[0][1] - top_langs[1][1]) < 0.3:
                logger.info("Detected potential code-mixing between Hindi and English")
                return "hi", True, top_langs[0][1]
        
        return top_langs[0][0], False, top_langs[0][1]
    except Exception as e:
        logger.error(f"Language detection error: {str(e)}")
        return "en", False, 1.0

def process_audio_chunk(chunk, model, language, is_mixed, prompt=""):
    """Process a single audio chunk."""
    try:
        # Ensure chunk is properly padded/trimmed
        chunk = whisper.pad_or_trim(chunk)
        
        # Transcribe chunk
        result = model.transcribe(
            chunk,
            language=language,
            initial_prompt=prompt,
            condition_on_previous_text=True,
            fp16=False
        )
        
        return result["text"].strip()
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        return ""

def transcribe_audio(audio_path, transcription_area=None, language=None):
    """Transcribe audio with improved memory handling and error recovery."""
    start_time = time.time()
    try:
        # Detect language if not provided
        if language is None:
            language, is_mixed, confidence = detect_language_confidence(audio_path)
            logger.info(f"Detected language: {language} (mixed: {is_mixed}, confidence: {confidence:.2f})")
        else:
            is_mixed = language == "hi-en"
            if is_mixed:
                language = "hi"
        
        # Update UI with language detection
        if transcription_area:
            status = transcription_area.empty()
            status.info(f"Detected {'Hinglish' if is_mixed else language.upper()} content")
        
        # Initialize progress tracking
        if transcription_area:
            progress_bar = transcription_area.progress(0)
            progress_text = transcription_area.empty()
            output_area = transcription_area.empty()
        
        # Load model
        model = get_whisper_model("large", device="cpu")
        
        # Get appropriate prompt
        prompt = HINGLISH_PROMPT if is_mixed else (HINDI_PROMPT if language == "hi" else "")
        
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

import whisper
import streamlit as st
import logging
import os
import torch
import re
import librosa
import numpy as np
from pydub import AudioSegment
import time
from multiprocessing import Pool, cpu_count
import concurrent.futures
import gc
import psutil
from functools import lru_cache

logger = logging.getLogger(__name__)

def preprocess_audio(audio, sr):
    """Apply faster preprocessing to audio for transcription quality."""
    import librosa
    from scipy.signal import firwin, lfilter
    
    start_time = time.time()
    
    # Normalize and trim silence in one pass (faster)
    try:
        # Normalize
        audio = librosa.util.normalize(audio)
        
        # Trim silence from beginning and end (faster than full silence detection)
        audio = librosa.effects.trim(audio, top_db=20)[0]
        
        # Simple high-pass filter (faster than butter)
        nyq = sr / 2
        cutoff = 100 / nyq
        fir_coeff = firwin(101, cutoff, window='hamming')
        audio = lfilter(fir_coeff, [1.0], audio)
        
        # Simple dynamic range compression (faster method)
        # Use numpy vectorized operations for speed
        threshold = 0.05
        ratio = 2.0
        abs_audio = np.abs(audio)
        mask = abs_audio > threshold
        audio[mask] = np.sign(audio[mask]) * (threshold + (abs_audio[mask] - threshold) / ratio)
        
        logger.info(f"Audio preprocessing completed in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.warning(f"Error in audio preprocessing (using original): {str(e)}")
    
    return audio

def get_optimal_workers():
    """Determine the optimal number of worker processes based on system resources."""
    try:
        # Get system memory information
        mem = psutil.virtual_memory()
        available_memory_gb = mem.available / (1024 ** 3)
        
        # Get CPU count, but ensure we don't create too many workers
        cpu_cores = cpu_count()
        
        # Allocate workers based on available memory and cores
        # Each worker might need around 1-2GB for the Whisper model
        max_by_memory = max(1, int(available_memory_gb / 2))
        optimal_workers = min(cpu_cores - 1, max_by_memory)
        
        # Ensure at least 1 worker, but no more than 6 (diminishing returns beyond this)
        return max(1, min(optimal_workers, 6))
    except Exception as e:
        logger.warning(f"Error determining optimal workers: {str(e)}, using 2")
        return 2  # Default to 2 workers if detection fails

def split_audio(audio_path, chunk_length=60):  # 1 minute per chunk for faster processing
    """Split audio into chunks of specified length (in seconds)."""
    try:
        # For large files, use incremental loading to reduce memory usage
        try:
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            logger.info(f"Audio file size: {file_size_mb:.1f} MB")
            
            if file_size_mb > 100:  # If file is larger than 100MB
                logger.info("Large file detected, using memory-optimized loading")
                # Use lower quality for very large files to save memory
                audio, sr = librosa.load(audio_path, sr=16000, mono=True, res_type='kaiser_fast')
            else:
                audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            logger.warning(f"Error checking file size: {str(e)}")
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Apply preprocessing to improve audio quality
        audio = preprocess_audio(audio, sr)
        total_length = len(audio) / sr  # Total length in seconds
        chunk_samples = int(chunk_length * sr)  # Samples per chunk

        chunks = []
        temp_files = []
        for start in range(0, len(audio), chunk_samples):
            end = min(start + chunk_samples, len(audio))          
            chunk = audio[start:end]
            temp_file = f"temp_chunk_{start//chunk_samples}.wav"
            audio_seg = AudioSegment(
                (chunk * 32767).astype(np.int16).tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=1
            )
            audio_seg.export(temp_file, format="wav")
            chunks.append(temp_file)
            temp_files.append(temp_file)
        logger.info("Split audio into %d chunks", len(chunks))
        return chunks, temp_files, total_length
    except Exception as e:
        logger.error("Error splitting audio: %s", str(e))
        raise

def detect_language(audio_path):
    """Detect the language of the audio using whisper."""
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        device = "cpu"  # Force CPU usage for better compatibility
        model = get_whisper_model("small", device)  # Use cached small model for faster processing
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(device)
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        confidence = probs[detected_language]
        
        # If confidence is low, default to English
        if confidence < 0.5:
            logger.warning(f"Low confidence ({confidence:.2f}) for detected language: {detected_language}. Defaulting to English.")
            detected_language = "en"
        else:
            logger.info(f"Detected language: {detected_language} (confidence: {confidence:.2f})")
        return detected_language
    except Exception as e:
        logger.error("Error detecting language: %s", str(e))
        raise

def clean_transcription(text):
    """Clean up the transcription text to remove repetitions and improve quality."""
    # First, normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove excessive repetitions (3 or more repetitions of patterns longer than 2 chars)
    text = re.sub(r'(.{3,}?)\1{2,}', r'\1', text)
    
    # Remove repeated Hindi characters (like अअअअ becomes अ)
    text = re.sub(r'([अ-ह])\1{2,}', r'\1', text)
    
    # Clean up multiple punctuation
    text = re.sub(r'[ ,।.!?]{2,}', ' ', text)
    
    # Clean repetitive words (like "तो तो तो" becomes "तो")
    words = text.split()
    cleaned_words = []
    prev_word = None
    repeat_count = 0
    
    for word in words:
        if word == prev_word:
            repeat_count += 1
            if repeat_count < 2:  # Allow at most one repetition
                cleaned_words.append(word)
        else:
            cleaned_words.append(word)
            prev_word = word
            repeat_count = 0
    
    text = ' '.join(cleaned_words)
    
    # Fix punctuation for better sentence structure
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    cleaned_sentences = []
    seen = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        # Skip empty or very short sentences
        if len(sentence) < 5:
            continue
            
        # Only add if not a duplicate (case-insensitive check)
        sentence_lower = sentence.lower()
        if sentence_lower not in seen:
            if not sentence.endswith(('.', '!', '?', '।')):
                if re.search(r'[अ-ह]', sentence):  # Hindi text
                    sentence += '।'
                else:
                    sentence += '.'
            cleaned_sentences.append(sentence)
            seen.add(sentence_lower)
            
    return ' '.join(cleaned_sentences)

@lru_cache(maxsize=1)
def get_whisper_model(model_name="small", device="cpu"):
    """Cache the model to prevent repeated loading in the same process."""
    logger.info(f"Loading Whisper {model_name} model (will be cached)")
    return whisper.load_model(model_name, device=device)

def process_chunk(chunk_info):
    """Process a single audio chunk with improved quality."""
    chunk_path, language = chunk_info
    try:
        # Use cached model to prevent repeated loading
        device = "cpu"
        model = get_whisper_model("small", device)
        
        # Set appropriate initial prompt based on language
        initial_prompt = ""
        if language == "hi":
            initial_prompt = "ये एक शिक्षा से संबंधित वीडियो है। इसमें किसी शिक्षक द्वारा छात्रों को पढ़ाया जा रहा है।"
        elif language == "en":
            initial_prompt = "This is an educational video with a teacher explaining concepts to students."
        
        # Use optimized parameters for better quality
        result = model.transcribe(
            chunk_path,
            language=language,
            fp16=False,
            temperature=0.0,
            initial_prompt=initial_prompt,  # Provide context hint
            condition_on_previous_text=True,
            word_timestamps=True,  # Enable word-level timestamps for better segmentation
            verbose=False  # Reduce logging for speed
        )
        
        # Clean transcription before returning
        cleaned_text = clean_transcription(result["text"].strip())
        return cleaned_text
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_path}: {str(e)}")
        return ""

def transcribe_audio(audio_path, transcription_area=None, language="en"):
    """Transcribe audio using whisper with specified language - optimized for speed."""
    start_time = time.time()
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Split audio into smaller chunks for faster processing
        chunks, temp_files, total_length = split_audio(audio_path, chunk_length=60)
        total_chunks = len(chunks)
        logger.info(f"Split {total_length:.1f}s audio into {total_chunks} chunks of 60s each")

        # Set up improved UI with estimated time
        if transcription_area:
            col1, col2 = transcription_area.columns(2)
            progress_text = col1.empty()
            time_estimate = col2.empty()
            progress_bar = transcription_area.progress(0)
            text_area = transcription_area.empty()
            status = transcription_area.empty()
            
            # Initial information
            progress_text.text(f"Preparing transcription (0/{total_chunks})")
            time_estimate.text("Estimating time...")
            status.info(f"Processing {total_length:.1f} seconds of audio")
        # Decide on processing approach based on number of chunks
        if total_chunks <= 4:
            # For a small number of chunks, use ThreadPoolExecutor (simpler than multiprocessing)
            optimal_workers = min(total_chunks, get_optimal_workers())
            logger.info(f"Using thread pool with {optimal_workers} workers")
            transcriptions = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                # Create a CPU-only model for each thread to use (cached)
                device = "cpu"
                model = get_whisper_model("small", device)
                
                # Submit all tasks and track futures
                future_to_idx = {}
                for i, chunk_path in enumerate(chunks):
                    future = executor.submit(
                        model.transcribe,
                        chunk_path,
                        language=language,
                        fp16=False,
                        temperature=0.0,
                        condition_on_previous_text=True,
                        verbose=False
                    )
                    future_to_idx[future] = i
                
                # Process results as they complete
                completed = 0
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    completed += 1
                    
                    try:
                        result = future.result()
                        # Store with original index to preserve order
                        transcriptions.append((idx, result["text"].strip()))
                        
                        # Update progress
                        if transcription_area:
                            progress = completed / total_chunks
                            progress_bar.progress(progress)
                            elapsed = time.time() - start_time
                            rate = elapsed / completed
                            remaining = rate * (total_chunks - completed)
                            progress_text.text(f"Transcribing chunk {completed}/{total_chunks}")
                            time_estimate.text(f"Est. time remaining: {int(remaining)}s")
                            
                            # Show partial results occasionally
                            if completed % 2 == 0 or completed == total_chunks:
                                partial_results = sorted(transcriptions)
                                partial_text = " ".join([text for _, text in partial_results])
                                text_area.text_area(
                                    "Transcription in progress...",
                                    value=partial_text,
                                    height=300
                                )
                    except Exception as e:
                        logger.error(f"Error processing chunk {idx}: {str(e)}")
        else:
            # For more chunks, use multiprocessing
            logger.info("Using multiprocessing pool for parallel transcription")
            
            if transcription_area:
                status.info(f"Starting parallel processing with multiple CPU cores")
            
            # Prepare chunk data for processing
            chunk_data = [(chunk, language) for chunk in chunks]
            
            # Process in parallel (each process loads its own model)
            try:
                # Use 'spawn' for macOS compatibility
                # context = multiprocessing.get_context('spawn')
                optimal_processes = get_optimal_workers()
                logger.info(f"Using process pool with {optimal_processes} workers")
                with Pool(processes=optimal_processes) as pool:
                    # Show initial progress
                    if transcription_area:
                        progress_bar.progress(0.1)
                        progress_text.text("Processing chunks in parallel...")
                    
                    # Process chunks and gather results
                    results = pool.map(process_chunk, chunk_data)
                    transcriptions = [(i, text) for i, text in enumerate(results)]
            except Exception as e:
                logger.error(f"Parallel processing error: {str(e)}")
                # Fall back to sequential processing if parallel fails
                logger.info("Falling back to sequential processing")
                if transcription_area:
                    status.warning("Parallel processing failed, switching to sequential mode")
                
                transcriptions = []
                device = "cpu"
                model = get_whisper_model("small", device)
                
                # Process chunks sequentially
                for i, chunk_path in enumerate(chunks):
                    try:
                        result = model.transcribe(
                            chunk_path,
                            language=language,
                            fp16=False,
                            temperature=0.0,
                            condition_on_previous_text=True
                        )
                        transcriptions.append((i, result["text"].strip()))
                        
                        # Update progress
                        if transcription_area:
                            progress = (i + 1) / total_chunks
                            progress_bar.progress(progress)
                            progress_text.text(f"Transcribing... {progress * 100:.1f}%")
                    except Exception as e:
                        logger.error(f"Error processing chunk {i}: {str(e)}")
        
        # Combine transcriptions in the correct order
        sorted_transcriptions = sorted(transcriptions, key=lambda x: x[0])
        transcription = " ".join([text for _, text in sorted_transcriptions])
        
        # Force garbage collection to free memory
        gc.collect()

        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        # Apply a more thorough cleaning
        transcription = clean_transcription(transcription)
        
        # Additional post-processing to improve readability
        # 1. Remove repetitive sentences (common in whisper output)
        from collections import Counter
        sentences = transcription.split('.')
        unique_sentences = []
        sentence_counter = Counter()
        
        for sentence in sentences:
            clean_sentence = sentence.strip().lower()
            if clean_sentence and sentence_counter[clean_sentence] < 2:  # Allow up to 2 occurrences
                unique_sentences.append(sentence.strip())
                sentence_counter[clean_sentence] += 1
        
        transcription = '. '.join(unique_sentences)
        if not transcription.endswith('.'):
            transcription += '.'
        
        # Calculate processing statistics
        total_time = time.time() - start_time
        speed_factor = total_length / total_time if total_time > 0 else 0
        logger.info(f"Transcription completed in {total_time:.2f}s ({speed_factor:.2f}x real-time)")
        
        # Update UI with final results
        if transcription_area:
            text_area.text_area(
                "Final Transcription",
                value=transcription,
                height=300
            )
            progress_bar.progress(1.0)
            progress_text.text("Transcription complete!")
            time_estimate.text(f"Processed in {total_time:.1f}s")
            status.success(f"✓ Processed {total_length:.1f}s of audio at {speed_factor:.1f}x real-time")

        logger.info("Final transcription: %s", transcription)
        return transcription
    except Exception as e:
        logger.error("Error transcribing audio: %s", str(e))
        if transcription_area:
            transcription_area.error(f"Error transcribing audio: {str(e)}")
        return None
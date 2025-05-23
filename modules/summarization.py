from transformers import pipeline, MBartForConditionalGeneration, BartTokenizer, MBartTokenizer, BartForConditionalGeneration
import logging
import torch
import re
import string
from langdetect import detect, LangDetectException, DetectorFactory
import numpy as np
import collections
import gc
import os

# Set a consistent seed for langdetect for reproducible results
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

def get_script_statistics(text):
    """Get statistics about scripts used in the text."""
    # Define script ranges
    scripts = {
        'devanagari': re.compile(r'[\u0900-\u097F]'),
        'latin': re.compile(r'[a-zA-Z]'),
        'arabic': re.compile(r'[\u0600-\u06FF]'),
        'bengali': re.compile(r'[\u0980-\u09FF]'),
        'gurmukhi': re.compile(r'[\u0A00-\u0A7F]'),
        'tamil': re.compile(r'[\u0B80-\u0BFF]'),
        'telugu': re.compile(r'[\u0C00-\u0C7F]')
    }
    
    # Count characters in each script
    stats = {}
    total_chars = 0
    
    for script_name, pattern in scripts.items():
        matches = pattern.findall(text)
        count = len(matches)
        stats[script_name] = count
        total_chars += count
    
    # Convert counts to percentages if there are characters
    if total_chars > 0:
        for script in stats:
            stats[script] = (stats[script] / total_chars) * 100
    
    return stats

def detect_hindi_markers(text):
    """Detect common Hindi words and phrases in Latin script."""
    # Normalize text for better matching
    text = " " + text.lower() + " "
    
    # Common Hindi words in Latin script with spaces on both sides to avoid partial matches
    hindi_markers = [
        " hai ", " nahi ", " aur ", " kya ", " mai ", " hum ", " tum ", " aap ",
        " mera ", " tera ", " uska ", " hamara ", " tumhara ", " unka ",
        " ko ", " se ", " par ", " me ", " ek ", " do ", " teen ", " char ",
        " ka ", " ki ", " ke ", " bhi ", " to ", " jaise ", " kaise ",
        " achha ", " theek ", " bahut ", " kuch ", " sab ", " lekin "
    ]
    
    # Count occurrences of markers
    marker_count = sum(1 for marker in hindi_markers if marker in text)
    
    # Get marker density per 100 words
    word_count = len(text.split())
    if word_count > 0:
        marker_density = (marker_count / word_count) * 100
    else:
        marker_density = 0
        
    return {
        'count': marker_count,
        'density': marker_density
    }

def detect_language_with_fallbacks(text):
    """Detect language with multiple fallback options."""
    # Strip excessive whitespace and ensure minimum text size
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) < 10:  # If text is too short, default to English
        return "en"
    
    # Sample from multiple parts of long texts for more reliable detection
    sample_text = text
    if len(text) > 1000:
        # Take the first 300, middle 300, and last 300 characters
        middle_start = max(0, (len(text) // 2) - 150)
        samples = [
            text[:300],
            text[middle_start:middle_start+300],
            text[-300:]
        ]
        sample_text = " ".join(samples)
    
    # Get script statistics
    script_stats = get_script_statistics(sample_text)
    
    # Check for strong Indic script presence
    indic_scripts = ['devanagari', 'bengali', 'gurmukhi', 'tamil', 'telugu']
    total_indic = sum(script_stats.get(script, 0) for script in indic_scripts)
    
    # If dominant Indic script presence (>30%), return appropriate language
    if script_stats.get('devanagari', 0) > 30:
        logger.info(f"Detected Hindi based on script statistics: {script_stats['devanagari']:.1f}% Devanagari")
        return "hi"
    
    # First try langdetect
    try:
        lang = detect(sample_text)
        logger.info(f"Language detected by langdetect: {lang}")
        
        # Additional validation for Hindi detection
        if lang == "hi" or total_indic > 10:
            return "hi"
            
        # Check for Hindi in Latin script if langdetect returns English
        if lang == "en":
            hindi_markers = detect_hindi_markers(sample_text)
            if hindi_markers['density'] > 15:  # If more than 15% of words are Hindi markers
                logger.info(f"Detected Hindi in Latin script (marker density: {hindi_markers['density']:.1f}%)")
                return "hi"
                
        return lang
    except LangDetectException as e:
        logger.warning(f"langdetect failed: {str(e)}, falling back to script analysis")
    
    # If langdetect fails, use script-based detection
    if script_stats.get('devanagari', 0) > 5:
        logger.info(f"Detected Hindi from script statistics: {script_stats['devanagari']:.1f}% Devanagari")
        return "hi"
    
    # Check for Hindi in Latin script as last resort
    hindi_markers = detect_hindi_markers(sample_text)
    if hindi_markers['count'] > 5 and hindi_markers['density'] > 10:
        logger.info(f"Detected Hindi in Latin script (markers: {hindi_markers['count']}, density: {hindi_markers['density']:.1f}%)")
        return "hi"
    
    logger.info("Defaulting to English as language could not be confidently detected")
    return "en"

def is_code_mixed(text):
    """Detect if text contains significant code-mixing between languages."""
    # Get script statistics
    script_stats = get_script_statistics(text)
    
    # Check for significant presence of both Devanagari and Latin
    devanagari_percent = script_stats.get('devanagari', 0)
    latin_percent = script_stats.get('latin', 0)
    
    # If both scripts are present in significant amounts
    if devanagari_percent > 5 and latin_percent > 20:
        # Calculate script ratio only considering these two scripts
        total = devanagari_percent + latin_percent
        if total > 0:
            devanagari_ratio = devanagari_percent / total
            # If the ratio suggests mixed content
            if 0.15 <= devanagari_ratio <= 0.85:
                logger.info(f"Detected code-mixed text (Devanagari: {devanagari_percent:.1f}%, Latin: {latin_percent:.1f}%)")
                return True
    
    # Check for Hindi in Latin script (Hinglish)
    hindi_markers = detect_hindi_markers(text)
    
    # If significant Hindi markers are present in what appears to be mostly Latin text
    if hindi_markers['count'] >= 3 and hindi_markers['density'] >= 5:
        logger.info(f"Detected code-mixed text (Hindi markers in Latin script: {hindi_markers['count']} words, {hindi_markers['density']:.1f}% density)")
        return True
    
    # Analyze word patterns to detect language switching
    words = text.split()
    if len(words) >= 10:  # Only check if there's enough text
        # Look at character types in consecutive words to detect script switching
        prev_is_latin = None
        switches = 0
        
        for word in words:
            # Skip punctuation and numbers
            if not re.search(r'[a-zA-Z\u0900-\u097F]', word):
                continue
                
            # Check if word is predominantly Latin or Devanagari
            latin_chars = sum(1 for c in word if c in string.ascii_letters)
            devanagari_chars = len(re.findall(r'[\u0900-\u097F]', word))
            
            is_latin = latin_chars > devanagari_chars
            
            # Count script switches
            if prev_is_latin is not None and is_latin != prev_is_latin:
                switches += 1
                
            prev_is_latin = is_latin
        
        # Calculate switch density per 100 words
        switch_density = (switches / len(words)) * 100
        if switch_density > 5:  # More than 5% of words involve script switching
            logger.info(f"Detected code-mixed text (script switch density: {switch_density:.1f}%)")
            return True
    
    return False

def split_into_sentences(text, language="en"):
    """Split text into sentences, respecting natural sentence boundaries for different languages."""
    import re
    
    # Clean the text - replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    if language == "hi" or is_code_mixed(text):
        # Hindi/mixed-language sentence patterns
        # Handle Devanagari sentence endings (purna viram) and Latin punctuation
        # \u0964 is the Devanagari danda (purna viram)
        pattern = r'(?<=[.!?\u0964।])\s+'
        # Split and clean
        raw_sentences = re.split(pattern, text)
        sentences = []
        
        for sent in raw_sentences:
            sent = sent.strip()
            if sent:  # Only add non-empty sentences
                sentences.append(sent)
        
        # Handle cases where punctuation might be missing
        # Look for capital letters or Devanagari script after spaces as potential sentence boundaries
        if len(sentences) <= 1 and len(text) > 100:
            logger.info("Few sentence boundaries detected, trying alternate method")
            # Pattern for capital letters or Devanagari characters after space
            pattern = r'(?<=\s)(?=[A-Z\u0900-\u097F])'
            sentences = re.split(pattern, text)
            # Clean and filter sentences
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Only keep sentences with reasonable length
    else:
        # Standard English sentence splitting
        # Pattern to match sentence-ending punctuation followed by space and capital letter
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
    
    # Final cleaning
    sentences = [s.strip() for s in sentences if s.strip()]
    logger.info(f"Split text into {len(sentences)} sentences using {language} patterns")
    return sentences

def chunk_text_by_sentences(text, max_chunk_size=1000, language="en"):
    """Chunk text by preserving sentence boundaries."""
    sentences = split_into_sentences(text, language=language)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence exceeds the max chunk size and we already have content,
        # finish the current chunk and start a new one
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def calculate_summary_length(text_length):
    """Calculate appropriate summary length based on input text length."""
    # For very short text, use a high percentage
    if text_length < 500:
        return max(30, int(text_length * 0.7)), max(15, int(text_length * 0.3))
    # For medium text, use medium percentage
    elif text_length < 2000:
        return max(100, int(text_length * 0.4)), max(30, int(text_length * 0.15))
    # For long text, use lower percentage
    else:
        return max(150, int(text_length * 0.25)), max(50, int(text_length * 0.1))

def clean_transcript_for_summary(text):
    """Clean transcription text for better summarization."""
    # Remove repetitive introductory phrases (common in Whisper outputs)
    text = re.sub(r'This is (?:a |an )?(?:video|educational video) with a teacher explaining concepts to students\.?\s*', '', text)
    text = re.sub(r'Thank you\.?\s*', '', text)
    
    # Preserve important domain-specific terms
    important_terms = [
        r'IPS', r'police', r'officer', r'rank', r'Sam Brown', r'belt',
        r'airport', r'computer', r'science', r'technology',
        r'student', r'university', r'education'
    ]
    
    # Build a pattern that won't remove these terms even if repeated
    preserve_pattern = '|'.join(important_terms)
    
    # Normalize sentence boundaries
    text = re.sub(r'([.!?])\s*([A-Za-z])', r'\1 \2', text)
    
    # Clean up repeated phrases and words but preserve important terms
    def clean_repeats(match):
        if re.search(preserve_pattern, match.group(1), re.IGNORECASE):
            return match.group(0)  # Keep the repetition for important terms
        return match.group(1)  # Remove repetition for other words
    
    text = re.sub(r'(\b\w{3,}\b)(\s+\1\b)+', clean_repeats, text)
    
    # Fix common transcription errors
    text = re.sub(r'\s+I am\s+I am\s+', ' I am ', text)
    text = re.sub(r'\s+This is\s+This is\s+', ' This is ', text)
    
    # Split into sentences and remove duplicates while preserving order
    sentences = re.split(r'(?<=[.!?])\s+', text)
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        cleaned = sentence.strip().lower()
        # Only add if sentence has meaningful content and isn't a duplicate
        if cleaned and len(cleaned) > 15 and cleaned not in seen:
            # Check if sentence contains any important terms
            has_important_term = any(re.search(term, cleaned, re.IGNORECASE) for term in important_terms)
            if has_important_term or len(cleaned.split()) > 5:
                unique_sentences.append(sentence.strip())
                seen.add(cleaned)
    
    return ' '.join(unique_sentences)

def clean_and_normalize_text(text, language="en"):
    """Clean and normalize text for better summarization."""
    # First apply general transcript cleaning
    text = clean_transcript_for_summary(text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove repetitive patterns that might confuse the model
    text = re.sub(r'(.{3,}?)\1{2,}', r'\1', text)
    
    # Handle language-specific cleaning
    if language == "hi":
        # Clean up repeated Hindi characters
        text = re.sub(r'([अ-ह])\1{2,}', r'\1', text)
        
        # Add proper punctuation for Hindi
        if not text.endswith('।'):
            text += '।'
    
    return text

def split_text_with_context(text, max_chunk_size=1000, overlap=100):
    """Split text into chunks with overlapping context to maintain coherence."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        # If adding this sentence would exceed the chunk size and we already have content
        if current_length + sentence_length > max_chunk_size and current_chunk:
            # Join current chunk into text
            chunks.append(' '.join(current_chunk))
            
            # Create overlap by keeping some sentences for context
            overlap_sentences = []
            overlap_length = 0
            
            # Take the last few sentences that fit within overlap size
            for s in reversed(current_chunk):
                s_length = len(s.split())
                if overlap_length + s_length <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += s_length
                else:
                    break
            
            # Start new chunk with overlap sentences
            current_chunk = overlap_sentences
            current_length = overlap_length
        
        # Add current sentence to chunk
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add the last chunk if there's anything left
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def combine_summaries(summaries):
    """Combine multiple summaries while preserving key information."""
    # Remove duplicates while preserving order
    seen = set()
    unique_summaries = []
    
    for summary in summaries:
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        for sentence in sentences:
            cleaned = sentence.strip().lower()
            if cleaned and len(cleaned) > 10 and cleaned not in seen:
                unique_summaries.append(sentence.strip())
                seen.add(cleaned)
    
    # Join sentences with proper spacing
    combined = ' '.join(unique_summaries)
    
    # Final cleaning
    combined = re.sub(r'\s+', ' ', combined).strip()
    if not combined.endswith(('.', '!', '?')):
        combined += '.'
    
    return combined

def smart_text_chunking(text, language="en", max_chunk_size=800):
    """Improved chunking that respects sentence boundaries and handles multilingual text."""
    # First attempt to split by proper sentence endings
    if language == "hi":
        # For Hindi, use Devanagari danda and other punctuation
        sentence_pattern = r'(?<=[।.!?])\s+'
    else:
        # For English and other languages
        sentence_pattern = r'(?<=[.!?])\s+'
        
    sentences = re.split(sentence_pattern, text)
    
    # If we got only a few very long sentences, try simpler splitting
    if len(sentences) <= 3 and max(len(s) for s in sentences) > max_chunk_size:
        # For Hindi, split on common conjunctions
        if language == "hi":
            sentences = re.split(r'(?<=\s(और|लेकिन|परन्तु|किन्तु|क्योंकि|इसलिए))\s+', text)
        else:
            # For other languages, try splitting on common conjunctions
            sentences = re.split(r'(?<=\s(and|but|because|however|therefore))\s+', text)
            
    # If still long sentences, use character-based chunking as last resort
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            separator = " " if current_chunk else ""
            current_chunk += separator + sentence
    
    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    # If no chunks were created (very short text), use the original text
    if not chunks:
        chunks = [text]
        
    return chunks

def get_topic_markers(text):
    """Identify main topics in the text for better summary focus."""
    topics = {
        'police': r'\b(police|officer|rank|IPS|inspector|constable)\b',
        'education': r'\b(student|university|education|teacher|class)\b',
        'technology': r'\b(computer|science|technology|algorithm)\b',
        'transport': r'\b(airport|transport|vehicle|distance)\b'
    }
    
    topic_counts = {}
    for topic, pattern in topics.items():
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        topic_counts[topic] = matches
    
    # Return primary topics (those with significant mentions)
    main_topics = [topic for topic, count in topic_counts.items() if count > 2]
    logger.info(f"Detected main topics: {', '.join(main_topics) if main_topics else 'none specific'}")
    return main_topics

def apply_topic_focus(text, topics):
    """Apply topic-focused prompt to text for better summarization."""
    if not topics:
        return text
        
    topic_terms = []
    if 'police' in topics:
        topic_terms.extend(['police', 'rank', 'officer', 'IPS'])
    if 'education' in topics:
        topic_terms.extend(['student', 'university', 'education'])
    if 'technology' in topics:
        topic_terms.extend(['computer', 'science', 'technology'])
    if 'transport' in topics:
        topic_terms.extend(['airport', 'transport', 'distance'])
    
    # Add topic-focused prompt for better summarization
    prompt = f"Summarize this text focusing on {', '.join(topics)}:"
    return prompt + "\n" + text

def summarize_multilingual_text(text, language, max_length=None, min_length=None):
    """Improved multilingual summarization with proper language handling."""
    try:
        logger.info(f"Using mBART for multilingual summarization (language: {language})")
        
        # Use correct language codes for mBART
        src_lang = "hi_IN" if language == "hi" else "en_XX"
        
        # Initialize tokenizer and model
        model_name = "facebook/mbart-large-cc25"
        tokenizer = MBartTokenizer.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        model.to("cpu")  # Ensure CPU usage
        
        # Important: Set the source language properly
        tokenizer.src_lang = src_lang
        
        # Clean and preprocess text
        text = clean_and_normalize_text(text, language)
        
        # Calculate appropriate summary length
        if max_length is None or min_length is None:
            max_length, min_length = calculate_summary_length(len(text))
            logger.info(f"Auto-calculated summary length: max={max_length}, min={min_length}")
        
        # Split into smaller chunks with better boundaries
        chunks = smart_text_chunking(text, language)
        logger.info(f"Split text into {len(chunks)} chunks using smart chunking")
        
        summaries = []
        for chunk in chunks:
            # Skip empty chunks
            if not chunk.strip():
                continue
                
            # Tokenize
            inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
            inputs = inputs.to("cpu")
            
            # Generate summary without using forced_bos_token_id
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    num_beams=4,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    early_stopping=True
                )
            
            # Decode and add to summaries
            summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary_text)
        
        # Combine summaries
        combined_summary = " ".join(summaries)
        logger.info("Generated multilingual summary with mBART")
        return combined_summary
        
    except Exception as e:
        logger.error(f"mBART summarization failed: {str(e)}")
        # Fall back to extractive summarization for multilingual text
        logger.warning("Falling back to extractive summarization for multilingual content")
        return extractive_summarize(text, language=language)

def extractive_summarize(text, language="en", ratio=0.3):
    """Extractive summarization as fallback for multilingual text."""
    try:
        from nltk.tokenize import sent_tokenize
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Ensure NLTK resources are available
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            if language == "en":
                nltk.download('stopwords', quiet=True)
        except:
            pass
            
        # Split into sentences using language-aware function
        sentences = split_into_sentences(text, language=language)
        if len(sentences) <= 5:
            return text  # Return original text if it's already short
            
        # For multilingual/non-English text, avoid using English stopwords
        stop_words = 'english' if language == 'en' else None
        
        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(stop_words=stop_words)
        tfidf_matrix = tfidf.fit_transform(sentences)
        
        # Get sentence scores based on TF-IDF values
        sentence_scores = np.array([tfidf_matrix[i].sum() for i in range(len(sentences))])
        
        # Select top sentences (proportional to text length)
        num_sentences = max(5, int(len(sentences) * ratio))
        top_indices = sentence_scores.argsort()[-num_sentences:]
        top_indices = sorted(top_indices)  # Sort by position in text
        
        # Create summary from selected sentences
        summary = ' '.join([sentences[i] for i in top_indices])
        logger.info(f"Generated extractive summary for {language} text with {num_sentences} sentences")
        return summary
        
    except Exception as e:
        logger.error(f"Extractive summarization failed: {str(e)}")
        # If all else fails, return first few sentences
        sentences = split_into_sentences(text, language=language)
        summary_length = max(3, int(len(sentences) * 0.2))
        return ' '.join(sentences[:summary_length])

def calculate_dynamic_summary_length(text):
    """Calculate appropriate summary length based on content."""
    # Count words in text
    text_length = len(text.split())
    
    # For very short text, use a higher percentage
    if text_length < 500:
        max_length = max(200, min(text_length * 3 // 4, 512))
        min_length = max(50, max_length // 3)
    # For medium text
    elif text_length < 2000:
        max_length = max(300, min(text_length // 3, 768))
        min_length = max(100, max_length // 3)
    # For long text
    else:
        max_length = max(400, min(text_length // 4, 1024))
        min_length = max(150, max_length // 3)
    
    logger.info(f"Dynamic summary length: max={max_length}, min={min_length} for text of {text_length} words")
    return max_length, min_length

def get_optimal_device():
    """Determine the best available device for model inference"""
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon support
        device = "mps"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device} for model inference")
    return device


def cleanup_model_memory():
    """Clean up GPU memory after model usage"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        # Clean MPS memory on Apple Silicon
        torch.mps.empty_cache()


def summarize_text(text, language=None, max_length=None, min_length=None, progress_callback=None):
    """Enhanced summarization with better content preservation and topic awareness."""
    # First, detect the language if not provided
    try:
        # Update progress if callback provided
        if progress_callback:
            progress_callback(0.1, "Cleaning and preparing text...")
            
        # Clean text first to remove repetitive content
        text = clean_transcript_for_summary(text)
        
        # Identify main topics in the text
        topics = get_topic_markers(text)
        
        if progress_callback:
            progress_callback(0.2, "Analyzing language and content structure...")
            
        # Detect language and check for code-mixing if language not provided
        if language is None:
            language = detect_language_with_fallbacks(text)
        elif language == "hindi":
            language = "hi"  # Convert to language code
        elif language.lower() == "english":
            language = "en"  # Convert to language code
        
        # For non-English text, always check for code-mixing
        is_mixed = False
        if language != "en":
            is_mixed = is_code_mixed(text)
            logger.info(f"Detected language: {language}, code-mixed: {is_mixed}")
        
        # Use multilingual model for Hindi or mixed content
        if language == "hi" or is_mixed:
            logger.info(f"Using multilingual summarization for {language} text (code-mixed: {is_mixed})")
            return summarize_multilingual_text(text, language, max_length, min_length)
            
        # For English text, use enhanced BART processing
        logger.info("Using enhanced BART for English text summarization")
        
        # Calculate appropriate summary length based on content
        if max_length is None or min_length is None:
            max_length, min_length = calculate_dynamic_summary_length(text)
            
        # Use BART with optimized parameters
        try:
            # Get optimal device based on availability
            device = get_optimal_device()
            
            try:
                # Load model components separately for better control
                model_name = "facebook/bart-large-cnn"
                
                # Load tokenizer and model with explicit settings
                tokenizer = BartTokenizer.from_pretrained(model_name)
                model = BartForConditionalGeneration.from_pretrained(model_name)
                
                # Move model to appropriate device
                model = model.to(device)
                
                # Create pipeline with explicit model and tokenizer
                bart_summarizer = pipeline(
                    "summarization",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if device == "cuda" else device,  # Format properly for pipeline
                )
                
                logger.info(f"Successfully loaded BART model on {device}")
                
            except Exception as model_error:
                logger.warning(f"Error loading BART model components: {str(model_error)}")
                logger.info("Falling back to simpler pipeline initialization")
                
                # Fallback to simpler pipeline initialization
                bart_summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if device == "cuda" else -1 if device == "cpu" else device
                )
            
            # Calculate token ratio for this model (words to tokens)
            token_ratio = 1.3  # Typical ratio for BART tokenizer
            
            # Adjust max_length to be in tokens instead of words
            token_max_length = int(max_length * token_ratio) if max_length else 1024
            token_min_length = int(min_length * token_ratio) if min_length else 150
            
            # Ensure token lengths are within model limits
            token_max_length = min(token_max_length, 1024)  # BART's maximum is around 1024
            
            # Set only max_new_tokens, not both max_length and max_new_tokens
            generation_kwargs = {
                "max_new_tokens": token_max_length,  # Use max_new_tokens instead of max_length
                "min_new_tokens": token_min_length,
                "do_sample": False,
                "num_beams": 4,           # Use beam search for better quality
                "length_penalty": 2.0,    # Encourage longer summaries
                "early_stopping": True,
            }
            
            # Split into smaller chunks while preserving context
            chunks = split_text_with_context(text, max_chunk_size=1000, overlap=150)
            logger.info(f"Split text into {len(chunks)} chunks with overlapping context")
            
            summaries = []
            chunk_count = 0
            total_chunks = len(chunks)
            
            for chunk in chunks:
                chunk_count += 1
                
                # Update progress
                if progress_callback:
                    progress = 0.2 + (0.6 * chunk_count / total_chunks)
                    progress_callback(progress, f"Summarizing chunk {chunk_count} of {total_chunks}")
                
                # Skip very short chunks
                if len(chunk.split()) < 50:
                    continue
                
                # Apply topic focus to this chunk if topics were detected
                if topics:
                    chunk_with_focus = apply_topic_focus(chunk, topics)
                else:
                    chunk_with_focus = chunk
                    
                # Generate summary for this chunk
                summary = bart_summarizer(
                    chunk_with_focus,
                    **generation_kwargs
                )
                summaries.append(summary[0]["summary_text"])
                
                # Clean up memory after each chunk
                if chunk_count % 2 == 0:  # Every two chunks
                    cleanup_model_memory()
            
            # Combine summaries intelligently
            final_summary = combine_summaries(summaries)
            logger.info("Enhanced summary generated via BART")
            
            # Final cleanup
            cleanup_model_memory()
            
            # Update progress
            if progress_callback:
                progress_callback(0.9, "Finalizing summary")
                
            return final_summary
        except Exception as e:
            logger.warning(f"BART summarization failed: {str(e)}. Falling back to T5.")
            try:
                # Get device for T5
                device = get_optimal_device()
                print(f"Device set to use {device}")
                
                # Fallback to T5-base with enhanced parameters
                t5_summarizer = pipeline(
                    "summarization",
                    model="t5-base",
                    device=0 if device == "cuda" else -1 if device == "cpu" else device
                )
                
                # Use same chunks with context preservation
                summaries = []
                for chunk in chunks:
                    # Skip very short chunks
                    if len(chunk.split()) < 50:
                        continue
                    
                    # Apply topic focus to this chunk if topics were detected
                    if topics:
                        chunk_with_focus = apply_topic_focus(chunk, topics)
                    else:
                        chunk_with_focus = chunk
                        
                    # Generate summary with T5
                    # T5 token ratio is different
                    t5_token_ratio = 1.5
                    
                    # Adjust max_length for T5
                    t5_max_tokens = int(max_length * t5_token_ratio) if max_length else 512
                    t5_min_tokens = int(min_length * t5_token_ratio) if min_length else 100
                    
                    # Use max_new_tokens for T5 as well
                    t5_generation_kwargs = {
                        "max_new_tokens": min(t5_max_tokens, 512),
                        "min_new_tokens": min(t5_min_tokens, 100),
                        "do_sample": False,
                        "num_beams": 4,
                        "length_penalty": 2.0,
                        "early_stopping": True
                    }
                    
                    summary = t5_summarizer(
                        chunk_with_focus,
                        **t5_generation_kwargs
                    )
                    summaries.append(summary[0]["summary_text"])
                
                # Use the same intelligent combination method
                final_summary = combine_summaries(summaries)
                logger.info("Summary generated via T5-base fallback")
                
                # Clean up T5 model memory
                cleanup_model_memory()
                
                return final_summary
            except Exception as e2:
                logger.error("T5 summarization failed: %s", str(e2))
                if "sentencepiece" in str(e2).lower():
                    return "Error: sentencepiece is required for transformers models. Install with 'brew install sentencepiece' and 'pip install sentencepiece'."
                
                # Fall back to extractive summarization as last resort
                logger.warning("Falling back to extractive summarization")
                return extractive_summarize(text)
    except Exception as e:
        logger.error("All summarization methods failed: %s", str(e))
        return f"Error summarizing text: {str(e)}"

# Expose summarize_text as generate_summary for external imports
generate_summary = summarize_text

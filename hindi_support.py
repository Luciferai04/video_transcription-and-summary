def detect_language_with_fallbacks(text):
    """Enhanced language detection with better Hindi and Hinglish support."""
    # Check for strong Devanagari presence first
    devanagari_chars = len(re.findall(r"[ऀ-ॿ]", text))
    total_chars = len(text)
    
    if devanagari_chars / total_chars > 0.3:
        logger.info("Detected Hindi based on Devanagari script presence")
        return "hi", True
    
    # Check for Hinglish markers
    hinglish_patterns = [
        r"(hai|hain|ka|ki|ko|me|mai|mera|tera|kya|aur|nahi|nai|kar|karo|raha|rhe|wala|naam)",
        r"(matlab|matlab ke|samjho|samajh|bol|bolo|dekh|dekho|suno|batao|pata|chalte|chalo)",
        r"(bohot|bahut|thoda|zyada|kam|jyada|accha|theek|sahi|galat|pura|poora|wahi|yehi)"
    ]
    
    hinglish_markers = 0
    for pattern in hinglish_patterns:
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        hinglish_markers += matches
    
    words = text.split()
    hinglish_density = hinglish_markers / len(words) if words else 0
    
    if hinglish_density > 0.15:
        logger.info(f"Detected Hinglish based on marker density: {hinglish_density:.2f}")
        return "hi", True
    
    # Try langdetect with better error handling
    try:
        lang = detect(text)
        if lang == "hi":
            return "hi", True
        return lang, False
    except:
        # Default to English if detection fails
        logger.warning("Language detection failed, defaulting to English")
        return "en", False

def clean_hindi_text(text):
    """Clean and normalize Hindi and Hinglish text."""
    # Remove repeated words (both Devanagari and Latin script)
    text = re.sub(r"([ऀ-ॿ]+)(\s+\1)+", r"\1", text)  # Hindi
    text = re.sub(r"([a-zA-Z]+)(\s+\1)+", r"\1", text)  # English
    
    # Fix common Hinglish patterns
    text = re.sub(r"(k|ke)\s+(liye|liy|lia)", "ke liye", text, flags=re.IGNORECASE)
    text = re.sub(r"(h|he|hai|hain)", "hai", text, flags=re.IGNORECASE)
    text = re.sub(r"(kr|kar|kro)", "kar", text, flags=re.IGNORECASE)
    
    # Fix spacing around Hindi punctuation
    text = re.sub(r"\s*।\s*", "। ", text)
    text = re.sub(r"\s*\|\s*", "। ", text)
    
    # Ensure proper sentence endings
    if not text.strip().endswith(("।", ".", "!", "?")):
        text = text.strip() + "।"
    
    return text.strip()

def summarize_hinglish_text(text, max_length=None, min_length=None):
    """Specialized summarization for Hinglish content."""
    try:
        # Clean and normalize text
        text = clean_hindi_text(text)
        
        # Calculate appropriate lengths if not provided
        if max_length is None or min_length is None:
            text_length = len(text.split())
            max_length = min(512, max(200, text_length // 3))
            min_length = max(50, max_length // 3)
        
        # Use mBART for better multilingual handling
        model_name = "facebook/mbart-large-cc25"
        tokenizer = MBartTokenizer.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        
        # Set source language
        tokenizer.src_lang = "hi_IN"
        
        # Split text into manageable chunks
        chunk_size = 1000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        summaries = []
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(
                inputs["input_ids"],
                num_beams=4,
                length_penalty=2.0,
                max_length=max_length,
                min_length=min_length,
                early_stopping=True
            )
            
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        
        # Combine summaries
        final_summary = " ".join(summaries)
        final_summary = clean_hindi_text(final_summary)
        
        return final_summary
    except Exception as e:
        logger.error(f"Hinglish summarization failed: {str(e)}")
        return extractive_summarize(text, language="hi")

def summarize_text(text):
    """Enhanced text summarization with better Hindi and Hinglish support."""
    try:
        # Detect language and mixed content
        language, is_mixed = detect_language_with_fallbacks(text)
        
        # Clean text first
        text = clean_transcript_for_summary(text)
        
        if is_mixed:
            logger.info("Using specialized Hinglish summarization")
            return summarize_hinglish_text(text)
        elif language == "hi":
            logger.info("Using Hindi summarization")
            return summarize_multilingual_text(text, language="hi")
        else:
            logger.info("Using English summarization")
            return summarize_multilingual_text(text, language="en")
            
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        return extractive_summarize(text)

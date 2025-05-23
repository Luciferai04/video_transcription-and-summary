import re
import logging
from langdetect import detect

logger = logging.getLogger(__name__)

def detect_language_with_fallbacks(text):
    """Enhanced language detection with better Hindi and Hinglish support."""
    # Check for Devanagari presence (though we expect romanized output now)
    devanagari_chars = len(re.findall(r"[ऀ-ॿ]", text))
    total_chars = len(text)
    
    if devanagari_chars / total_chars > 0.3:
        logger.info("Detected Hindi based on Devanagari script presence")
        return "hi", True
    
    # Expanded Hinglish patterns for romanized Hindi detection
    hinglish_patterns = [
        # Common Hindi words in Roman script
        r"(hai|hain|ka|ki|ko|me|mai|mera|tera|kya|aur|nahi|nai|kar|karo|raha|rhe|wala|naam)",
        r"(matlab|matlab ke|samjho|samajh|bol|bolo|dekh|dekho|suno|batao|pata|chalte|chalo)",
        r"(bohot|bahut|thoda|zyada|kam|jyada|accha|theek|sahi|galat|pura|poora|wahi|yehi)",
        # Pronouns and demonstratives
        r"(main|mujhe|mujhko|hum|humko|humne|tu|tum|tumko|tumne|aap|aapko|aapne|yeh|ye|woh|wo|isse|usse)",
        # Verbs and verb forms
        r"(hona|karna|bolna|kehna|jata|gaya|gaye|gayi|kiya|kiye|kiyi|tha|the|thi|thi|ho|karenge|karke)",
        # Question words
        r"(kaun|kahan|kab|kyun|kaise|kitna|kitne|kitni)",
        # Conjunctions and connectors
        r"(aur|ya|lekin|magar|phir|phir bhi|kyunki|isliye|to|toh|agar)"
    ]
    
    hinglish_markers = 0
    for pattern in hinglish_patterns:
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        hinglish_markers += matches
    
    words = text.split()
    hinglish_density = hinglish_markers / len(words) if words else 0
    
    # Lower threshold for Hinglish detection since we're now focusing on romanized Hindi
    if hinglish_density > 0.10:
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
    """Clean and normalize romanized Hindi (Hinglish) text."""
    # Remove repeated words (Latin script mainly since we're using romanized text)
    text = re.sub(r"([a-zA-Z]+)(\s+\1)+", r"\1", text)
    
    # Fix common Hinglish spelling variations and patterns
    text = re.sub(r"(k|ke)\s+(liye|liy|lia|liyay)", "ke liye", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(h|he|hai|hain|hey|hein|hae)\b", "hai", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(kr|kar|kro|karo|krna|karna)\b", "kar", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(hy|hai|hay|haii)\b", "hai", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(nahee|nhi|nahin|nahi|nhi|nay|nai)\b", "nahi", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(mein|me|may|mei)\b", "main", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(hum|ham|hm)\b", "hum", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(ap|aap|aapko)\b", "aap", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(ky|kyu|kyun|kyon|kiyun)\b", "kyun", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(bht|bhut|bahut|bahot)\b", "bahut", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(thik|teek|theek|thek)\b", "theek", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(acha|accha|achchha|acchha)\b", "accha", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(hm|hmm|hmm+)\b", "hmm", text, flags=re.IGNORECASE)
    
    # Fix spacing and sentence structure
    text = re.sub(r"\s+", " ", text)  # Normalize spacing
    text = re.sub(r"\s*\.\s*", ". ", text)  # Fix period spacing
    
    # Ensure proper sentence endings
    if not text.strip().endswith((".", "!", "?")):
        text = text.strip() + "."
    
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
        
        # Set source language (using English since our input is now romanized)
        tokenizer.src_lang = "en_XX"
        
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

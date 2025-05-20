from transformers import pipeline
import logging
import torch

logger = logging.getLogger(__name__)

def generate_questions(text, num_questions=5):
    """Generate questions from text using T5-QA."""
    try:
        # Initialize T5-QA model for question generation
        question_generator = pipeline(
            "text2text-generation",
            model="valhalla/t5-base-qa-qg-hl",
            device=0 if torch.backends.mps.is_available() else -1
        )
        # Split text into chunks to handle long inputs
        chunk_size = 1000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        questions = []
        for chunk in chunks:
            # Generate questions for each chunk
            input_text = f"generate questions: {chunk}"
            generated = question_generator(
                input_text,
                max_length=50,
                num_return_sequences=min(num_questions - len(questions), 3),
                do_sample=True
            )
            questions.extend([q["generated_text"] for q in generated])
            if len(questions) >= num_questions:
                break
        questions = questions[:num_questions]
        logger.info("Questions generated via T5-QA")
        return questions if questions else ["No questions generated"]
    except Exception as e:
        logger.error("Error generating questions: %s", str(e))
        if "sentencepiece" in str(e).lower():
            return ["Error: sentencepiece is required for T5-QA. Install with 'brew install sentencepiece' and 'pip install sentencepiece'."]
        return [f"Error generating questions: {str(e)}"]

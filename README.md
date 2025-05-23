app.py – Main Video Transcription & Summarization API Server

app.py is the primary Flask-based API server for this project, designed to handle the full video-to-text pipeline for both uploaded video files and YouTube videos. It provides robust asynchronous job management, integration with SocketIO for real-time progress updates, and exposes rich endpoints for transcription, summarization, question generation, translation, and semantic chat.

Key Features:
•  Video Input: Accepts both direct uploads (mp4, mkv, avi, mov) via /upload and YouTube URLs via /youtube.
•  Audio Extraction & Enhancement: Uses FFmpeg and audio processing pipelines (librosa, soundfile, noisereduce) to extract and clean audio before transcription.
•  Automatic Language Detection: Detects language (English, Hindi, or Hinglish), with enhanced support for multilingual/code-mixed audio.
•  Speech-to-Text: Integrates with advanced ML models for transcription (e.g., OpenAI Whisper).
•  Text Summarization & Question Generation: NLP modules generate concise summaries and questions from the full transcript.
•  Real-time Progress: Uses Flask-SocketIO (gevent backend recommended) for real-time client updates on processing state.
•  Semantic Chat: Provides a /chat endpoint using sentence-transformers (or fallback approaches) to enable retrieval-augmented answers over the transcript.
•  Session Management: Maintains session state in memory, associating each job with a unique session_id.
•  API-First: Designed to be consumed via REST (cURL, Postman, or integrated UIs).
•  Production/Dev Flexibility: Can be run directly for development or deployed with Gunicorn+gevent for production, with support code in wsgi.py.

Usage:  
Ideal for building front-ends requiring accurate video transcription, educational platforms, or data annotation tools where asynchronous processing and rich post-processed output are critical.



app2.py – Alternate/Experimental API Server

app2.py is an alternate entry point for the same or similar backend workflow. It largely mirrors app.py in features and endpoints, acting as a drop-in replacement but with potential code structure tweaks, experiments, or development-stage enhancements.

Key Features:
•  Same Core Pipeline: Supports video file uploads, YouTube import, audio processing, transcription, summary, questions, and translation.
•  Flask-SocketIO Integration: Also supports WebSocket updates and session-based asynchronous progress.
•  Compatible API: Endpoints and data formats are kept synchronized with app.py so clients/Postman collections generally work unchanged.
•  Development/Ease-of-Use: May include simplifications or more rapid iteration of features, making it suitable for prototyping or quick testing without impacting the production codebase.
•  Flexible Startup: Intended mainly for local/development use (though can be run via Gunicorn if needed).

Usage:  
Best for testing new features, using a simplified Flask run loop, or for learners wanting to experiment with the codebase before moving to production.



Major Differences & Use Cases:

•  app.py: The main, production-ready server. Has more mature job management, logging, and is “battle-tested” for deployment behind Gunicorn+gevent. Use this if you want reliability and advanced features.
•  app2.py: Experimental or possibly simplified version. Use when iterating on new ideas, when running multiple instances for comparison, or as a less-coupled playground.

Both apps share dependencies, modules (in /modules/), and a similar API surface. The choice mostly depends on your purpose: production (app.py) vs. fast prototyping or experimentation (app2.py).

# Video Transcription and Summary – Main Server (`app.py` & `wsgi.py`)

## Overview

This application provides a full-featured video transcription and summary API server. It includes:
- Video upload (files or YouTube URLs)
- Automatic language detection (English, Hindi, Hinglish support)
- Audio extraction, silence removal, and enhanced processing
- Asynchronous job handling with WebSocket updates
- Text summary and question generation endpoints
- Easily used with Postman or similar API clients

---

## Prerequisites

- **Python:** 3.11 or higher
- **FFmpeg:** must be installed and accessible in `$PATH` ([ffmpeg.org](https://ffmpeg.org/download.html))
- **Virtual Environment** (recommended)
- **All dependencies installed:**  
  `requirements2.txt` (ensure it’s up-to-date with actual code)

---

## 1. Environment Setup

```bash
cd /Users/soumyajitghosh/Downloads/video_transcription-and-summary-main

# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements2.txt
```

---

## 2. Running the Server

### **A. Development / Local Testing**

Run the Flask-SocketIO application with hot reload:

```bash
python app.py --host 127.0.0.1 --port 5001
```

### **B. Production (Recommended): Gunicorn + Gevent WS**

Start the server using Gunicorn and wsgi.py for proper concurrency and WebSocket support:

```bash
gunicorn --worker-class gevent --workers 1 --bind 127.0.0.1:5001 wsgi:application
```

> **Tip:** Ensure nothing else is running on port 5001. To kill prior processes:
>
>     lsof -ti:5001 | xargs kill -9 2>/dev/null

---

## 3. API Endpoints

| Endpoint       | Method | Description                                          |
|----------------|--------|------------------------------------------------------|
| `/`            | GET    | Health check                                         |
| `/upload`      | POST   | Upload video file (multipart/form-data)              |
| `/youtube`     | POST   | Process YouTube URL (`{"url": "https://..."}`)       |
| `/transcribe`  | POST   | Begin transcription (`session_id` + `video_path`)    |
| `/status`      | GET    | Poll processing status (`session_id` as query param) |
| `/summary`     | GET    | Fetch transcription summary (`session_id`)           |
| `/questions`   | GET    | Fetch auto-generated questions (`session_id`)        |
| `/translate`   | POST   | Translate (for Hindi) (`session_id`)                 |
| `/chat`        | POST   | AI content chat interface (`session_id`, `message`)  |

#### Example: Health Check
```bash
curl http://127.0.0.1:5001/
```

#### Example: Upload File (with curl)
```bash
curl -F "file=@video.mp4" http://127.0.0.1:5001/upload
```

#### Example: YouTube Import
```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"url": "https://www.youtube.com/watch?v=XXXX"}' \
    http://127.0.0.1:5001/youtube
```

> **See full API details above or in `/README.md`.** All API requests can be tested using [Postman](https://www.postman.com/) or similar tools.

---

## 4. Troubleshooting & FAQ

- **Port Error**:  
  `OSError: [Errno 48] Address already in use`  
  → Stop any other processes running on the same port:  
    `lsof -ti:5001 | xargs kill -9`
- **FFmpeg not found errors**:  
  → Install FFmpeg (`brew install ffmpeg` on Mac, or see [official docs](https://ffmpeg.org/download.html))
- **WebSocket/SocketIO issues**:  
  → Ensure `gevent` and `gevent-websocket` are installed, and you are NOT using `eventlet`.
- **Dependency Problems**:  
  → Confirm you’ve installed all packages from `requirements2.txt` in your active virtual environment.
- **Other**:  
  - Check logs for stack traces
  - For long jobs, use `/status` to poll progress

---

## 5. Notes

- **Session Handling:**  
  Always use the `session_id` provided by upload or YouTube import for subsequent requests.
- **Uploads & Temp Files:**  
  Cleaned up automatically, but see logs for errors if disk usage spikes.
- **Security:**  
  Do NOT expose this development system to a public network without authentication/hardening.

---


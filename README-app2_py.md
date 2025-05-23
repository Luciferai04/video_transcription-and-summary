# Video Transcription and Summary – Alternate Server (`app2.py`)

## Introduction

`app2.py` is an alternate, full-featured Flask server for the same video transcription and summarization pipeline provided by `app.py`, intended for testing, experimentation, or as a simplified deployment method. All core endpoints are provided, with similar Postman/cURL API compatibility. It runs on your preferred port (default: 5002).

---

## Prerequisites

- **Python:** 3.11 or higher  
- **FFmpeg:** installed and in your `$PATH`
- **Virtual environment** recommended
- **Dependencies installed:**  
  via `requirements2.txt` (shared with main app)

---

## 1. Environment Setup

```bash
cd /Users/soumyajitghosh/Downloads/video_transcription-and-summary-main
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements2.txt
```

---

## 2. Running the Server

### **A. Standard/Development Mode**
Launch on any port (e.g., 5002):

```bash
python app2.py --host 127.0.0.1 --port 5002
# or just
python app2.py
```

### **B. Production/Concurrent Mode (OPTIONAL)**

You may also run this app with Gunicorn + gevent if needed:

```bash
gunicorn --worker-class gevent --workers 1 --bind 127.0.0.1:5002 app2:app
```

> *(Adjust the port as needed to avoid conflicts.)*

---

## 3. API Endpoints

`app2.py` exposes almost the same endpoints as the main server.

| Endpoint       | Method | Description
|----------------|--------|------------------------------------------------------|
| `/`            | GET    | Health check                                         |
| `/upload`      | POST   | Upload video file (multipart/form-data)              |
| `/youtube`     | POST   | Process YouTube URL (`{"url": "..."}`)               |
| `/transcribe`  | POST   | Begin transcription (`session_id`, `video_path`)     |
| `/status`      | GET    | Poll processing status (`session_id` as query param) |
| `/summary`     | GET    | Fetch transcription summary (`session_id`)           |
| `/questions`   | GET    | Get auto-generated questions (`session_id`)          |
| `/translate`   | POST   | Translate Hindi/Hinglish (`session_id`)              |
| `/chat`        | POST   | Chat with content (`session_id`, `message`)          |

**Example: CURL File Upload**  
```bash
curl -F "file=@video.mp4" http://127.0.0.1:5002/upload
```

**Example: Postman Use**  
- See `README-app_py.md` for Postman workflow and JSON body examples (substitute port 5002).

---

## 4. Troubleshooting

- **Port Already in Use**:  
  Stop any other services using your port:
  `lsof -ti:5002 | xargs kill -9`

- **ffmpeg not found**:  
  Ensure ffmpeg is installed and on your path.

- **Dependencies missing**:  
  Activate your venv and run:
  `pip install -r requirements2.txt`

- **WebSocket errors**:  
  Use `gevent` for production deployment (see above), and avoid `eventlet` on Python 3.11+.

---

## 5. Additional Notes

- `app2.py` is functionally similar to `app.py`, with minor implementation or experimental differences.
- Use your session IDs and video paths as returned by `/upload` or `/youtube` in subsequent API calls.
- Check logs and console for error messages and stack traces during operation.

---


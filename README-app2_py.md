# Video Transcription and Summary – Alternate UI/App (`app2.py`, Streamlit)

## Introduction

`app2.py` can operate as a **Streamlit web application** for the video transcription and summarization pipeline, offering a modern browser-based interface in addition to (or instead of) traditional Flask/REST endpoints.

- **Streamlit UI:** Easiest way for users to upload videos, process YouTube links, and interactively view results via a point-and-click web dashboard.
- **Flask API compatibility:** If explicitly run as a Flask server, can work with API tools like Postman (see below).
- **All major features:** Upload, language detection, full transcription, summary, questions, chat—accessible from the browser.

---

## Prerequisites

- **Python:** 3.11 or higher
- **FFmpeg:** must be installed and on your `$PATH`
- **Virtual environment:** recommended
- **Streamlit:** required (included in `requirements2.txt`)
- **Install dependencies:**
  ```bash
  pip install -r requirements2.txt
  ```
- *(All other dependencies from the main project are also required and provided via `requirements2.txt`.)*

---

## 1. Environment Setup

```bash
cd "system path name"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements2.txt
```

---

## 2. Running the Streamlit Web App (Recommended)

The easiest and most user-friendly way to use `app2.py` is via Streamlit UI.

```bash
streamlit run app2.py
```

- **Open your browser** to [http://localhost:8501](http://localhost:8501) (Streamlit’s default port).
- **Features available:** Video upload, YouTube import, progress indicators, transcription output, summaries, questions, chat—all via interactive web interface.

---

### Alternative: Running as an API Server

If you wish to use `app2.py` as a backend Flask API server (just like `app.py`), you may:

```bash
python app2.py --host 127.0.0.1 --port 5002
```
or for production:
```bash
gunicorn --worker-class gevent --workers 1 --bind 127.0.0.1:5002 app2:app
```
> *(Adjust port as needed to avoid conflicts)*

---

**Choosing the mode:**
- Use `streamlit run app2.py` for browser UI (recommended for most users).
- Use `python app2.py` or Gunicorn for Postman/cURL API access or development/testing.

---

## 3. Features Available in Streamlit UI

When launched with Streamlit, `app2.py` provides a clean web browser interface for:
- **Video Upload:** Browse and upload local video files for transcription.
- **YouTube Download:** Submit public YouTube links for direct processing.
- **Progress Feedback:** Real-time feedback while transcribing/converting jobs.
- **Language Detection:** Automatic, with support for Hindi, English, and Hinglish.
- **Transcription Output:** View or download the full text transcript directly in your browser.
- **Summarization & Q&A:** See generated summaries and sample questions.
- **Chat:** Ask questions about the content directly in the browser.

You access all features at: [http://localhost:8501](http://localhost:8501)

---


## 4. Troubleshooting

**Streamlit-Specific Issues:**
- **Port 8501 Already in Use:** Another Streamlit app or other service is running on this port.  
  - Kill with `lsof -ti:8501 | xargs kill -9`  
  - Or run with `streamlit run app2.py --server.port=8502` for another port.
- **Page not updating/blank:** Clear cache, try another browser, or restart the server.
- **Permission Errors / FFmpeg not found:**  
  - Ensure ffmpeg is installed (`ffmpeg -version`).
  - Confirm you are in your venv, and all dependencies installed.

**General Issues:**
- **Dependencies missing:**  
  Activate your venv and run: `pip install -r requirements2.txt`
- **WebSocket or API errors:**  
  If running in API mode, ensure you use `gevent` worker for Gunicorn and NOT eventlet, especially on Python 3.11+.
- **Session not found/data loss:**  
  Each app instance holds session data in RAM. Be sure to start new uploads after restarting the app.

---


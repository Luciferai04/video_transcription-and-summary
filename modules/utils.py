import os
import tempfile
import yt_dlp
import logging

logger = logging.getLogger(__name__)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location."""
    try:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        logger.info("Uploaded file saved: %s", temp_path)
        return temp_path
    except Exception as e:
        logger.error("Error saving uploaded file: %s", str(e))
        raise

def download_youtube_video(url):
    """Download a YouTube video to a temporary location."""
    try:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "youtube_video.mp4")
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": temp_path,
            "merge_output_format": "mp4",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logger.info("YouTube video downloaded: %s", temp_path)
        return temp_path
    except Exception as e:
        logger.error("Error downloading YouTube video: %s", str(e))
        raise

# Video Transcription and Summary (with Hinglish Support)

A powerful Flask-based web application that transcribes videos (both uploaded files and YouTube URLs), generates summaries, and supports Hinglish (Romanized Hindi) output for Hindi content.

## Features

- Video upload and YouTube URL support
- Audio extraction and processing
- Automatic language detection
- Transcription with special Hinglish support for Hindi content
- Text summarization
- Real-time progress updates via WebSocket
- Question generation from content
- Chat interface for content interaction

## Setup Instructions

### Prerequisites

```bash
- Python 3.11 or higher
- FFmpeg
- Virtual environment (recommended)
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd video_transcription-and-summary
```

2. Create and activate virtual environment:
```bash
python -m venv venv311
source venv311/bin/activate  # On Windows: venv311\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements_fixed.txt
```

4. Ensure FFmpeg is installed:
- MacOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`
- Windows: Download from ffmpeg.org and add to PATH

## Project Structure

The project is organized as follows:

```
video_transcription-and-summary/
├── app.py                 # Main Flask application server
├── upload_video.py        # Script for handling video uploads
├── transcription.py       # Core transcription functionality
├── modules/               # Additional functionality modules
│   ├── hindi_support.py   # Hinglish conversion support
│   ├── summarization.py   # Text summarization module
│   └── question_gen.py    # Question generation from content
├── uploads/               # Temporary storage for uploaded videos
├── downloads/             # Storage for processed files
├── templates/             # HTML templates for web interface
├── static/                # CSS, JavaScript, and other static assets
├── venv311/               # Virtual environment (not tracked in git)
├── requirements.txt       # Original package dependencies
└── requirements_fixed.txt # Complete list of dependencies
```

The `requirements_fixed.txt` file contains all necessary dependencies including:
- `flask`, `flask_socketio`, and `flask_cors` for the web server
- `yt_dlp` for YouTube video processing
- `whisper` and related packages for transcription
- Additional libraries for text processing and summarization

This structure ensures clean separation of concerns and makes the application modular and maintainable.

## Usage

1. Start the server:
```bash
python app.py
```

2. Access the application:
- Web Interface: Open `http://localhost:5000` in your browser
- API: Send requests to available endpoints

## API Endpoints

- `POST /upload`: Upload video file
- `POST /youtube`: Process YouTube URL
- `POST /transcribe`: Start transcription
- `GET /status`: Check processing status
- `GET /summary`: Get content summary
- `GET /questions`: Get generated questions
- `POST /translate`: Translate Hindi content
- `POST /chat`: Chat with content

## Special Features

### Hinglish Transcription

The system now outputs romanized text (Hinglish) instead of Devanagari script for Hindi content. For example:
- Input (Audio): "मैं आज बहुत खुश हूँ"
- Output (Text): "Main aaj bahut khush hoon"

This makes the content more accessible and easier to read for users who prefer Roman script.

## Common Issues and Solutions

1. **Installation Errors**
   - Error: `FFmpeg not found`
   - Solution: Install FFmpeg and ensure it's in system PATH

2. **Memory Issues**
   - Error: `OutOfMemoryError`
   - Solution: 
     - Reduce chunk size in transcription settings
     - Free up system memory
     - Use CPU instead of GPU for processing

3. **Transcription Errors**
   - Error: `Language detection failed`
   - Solution: Ensure audio quality is good and try with a different section

4. **File Upload Issues**
   - Error: `File too large`
   - Solution: Check MAX_CONTENT_LENGTH in configuration

5. **YouTube Download Issues**
   - Error: `Unable to download video`
   - Solution: 
     - Verify URL is accessible
     - Check internet connection
     - Ensure video is not private/restricted

## Technical Details

### Memory Management

The application uses chunked processing to handle large files efficiently:
- Audio is processed in 30-second chunks
- Automatic garbage collection between chunks
- Proper resource cleanup after processing

### Language Detection

- Uses Whisper's language detection
- Enhanced with custom Hindi/Hinglish detection patterns
- Supports mixed language content (code-switching)

### File Handling

- Temporary files are automatically cleaned up
- Uploads are secured and validated
- Downloads are managed efficiently

## Contributing

Feel free to open issues or submit pull requests for:
- Bug fixes
- Feature enhancements
- Documentation improvements
- Performance optimizations

## License

MIT License - See LICENSE file for details

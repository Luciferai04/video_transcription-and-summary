#!/bin/bash
export HOST=127.0.0.1
export PORT=5001
export DEBUG=False

# Kill any existing processes on port 5001
lsof -ti:5001 | xargs kill -9 2>/dev/null || true

# Start the server with gunicorn
gunicorn --worker-class gevent --workers 1 --bind $HOST:$PORT 'wsgi:application' --log-level info --daemon

# Wait a moment for the server to start
sleep 2

# Check if the server is running
if curl -s http://127.0.0.1:5001/ > /dev/null; then
    echo "Server is running at http://$HOST:$PORT"
    echo "You can now use Postman with the following endpoints:"
    echo ""
    echo "1. Health Check (GET):"
    echo "   http://$HOST:$PORT/"
    echo ""
    echo "2. Upload Video (POST):"
    echo "   http://$HOST:$PORT/upload"
    echo "   - Form Data: key='file', type=File"
    echo ""
    echo "3. YouTube URL (POST):"
    echo "   http://$HOST:$PORT/youtube"
    echo "   - JSON Body: {'url': 'YOUTUBE_URL'}"
    echo ""
    echo "4. Transcribe (POST):"
    echo "   http://$HOST:$PORT/transcribe"
    echo "   - JSON Body: {'session_id': 'ID', 'video_path': 'PATH', 'language': 'auto'}"
    echo ""
    echo "5. Check Status (GET):"
    echo "   http://$HOST:$PORT/status?session_id=YOUR_SESSION_ID"
    echo ""
    echo "6. Get Summary (GET):"
    echo "   http://$HOST:$PORT/summary?session_id=YOUR_SESSION_ID"
    echo ""
    echo "7. Get Questions (GET):"
    echo "   http://$HOST:$PORT/questions?session_id=YOUR_SESSION_ID"
    echo ""
    echo "8. Translate (POST):"
    echo "   http://$HOST:$PORT/translate"
    echo "   - JSON Body: {'session_id': 'YOUR_SESSION_ID'}"
else
    echo "Failed to start server"
    exit 1
fi

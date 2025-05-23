#!/bin/bash

echo "Setting up Video Transcription Environment..."

# Check if Python 3.11 or higher is installed
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$python_version < 3.11" | bc -l) )); then
    echo "Error: Python 3.11 or higher is required (found $python_version)"
    exit 1
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg not found. Installing FFmpeg..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install ffmpeg
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    else
        echo "Please install FFmpeg manually for your operating system"
        exit 1
    fi
fi

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements2.txt

# Verify key installations
echo "Verifying installations..."
python3 -c "
import sys
packages = [
    'flask',
    'flask_socketio',
    'gevent',
    'torch',
    'transformers',
    'librosa',
    'yt_dlp'
]
missing = []
for package in packages:
    try:
        __import__(package)
        print(f'✓ {package} installed successfully')
    except ImportError as e:
        missing.append(package)
        print(f'✗ {package} installation failed')
if missing:
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "Setup completed successfully!"
    echo "To start the server, run:"
    echo "source venv/bin/activate"
    echo "./start_server.sh"
else
    echo "Setup failed. Please check the error messages above."
    exit 1
fi

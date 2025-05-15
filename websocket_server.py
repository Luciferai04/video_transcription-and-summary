from flask import Flask
from flask_sockets import Sockets
from faster_whisper import WhisperModel
import torch
import numpy as np
import logging
import io
import wave
import json
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

app = Flask(__name__)
sockets = Sockets(app)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize faster-whisper model
model = WhisperModel(
    "distil-large-v3",
    device="mps" if torch.backends.mps.is_available() else "cpu",
    compute_type="float16"
)

# Initialize the VAD model with torch.hub
try:
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    get_speech_timestamps = utils[0]
    logger.info("Successfully loaded silero-vad model and utils")
except Exception as e:
    logger.error(f"Failed to load silero-vad: {str(e)}")
    vad_model = None
    def get_speech_timestamps(audio_array, model, **kwargs):
        return [{'start': 0, 'end': len(audio_array)}]
    logger.warning("Using fallback VAD implementation")

def process_audio_chunk(audio_data, sample_rate=16000):
    """Process audio chunk with VAD and faster-whisper."""
    try:
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0

        speech_timestamps = get_speech_timestamps(
            audio,
            vad_model,
            sampling_rate=sample_rate,
            threshold=0.8,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300
        )

        if not speech_timestamps:
            logger.info("No speech detected")
            return None

        # Save detected audio to WAV in memory
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            audio_int16 = (audio * 32767.0).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())

        audio_buffer.seek(0)

        # Transcribe using Whisper
        segments, _ = model.transcribe(audio_buffer, language="en")
        result = " ".join([seg.text for seg in segments])
        return result
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return None

@sockets.route("/transcribe")
def transcribe_socket(ws):
    """Handle WebSocket connections for transcription."""
    while not ws.closed:
        try:
            audio_data = ws.receive()
            if audio_data:
                transcription = process_audio_chunk(audio_data)
                if transcription:
                    ws.send(json.dumps({"type": "transcription", "text": transcription}))
                    logger.info("Sent transcription: %s", transcription)
        except Exception as e:
            logger.error("WebSocket error: %s", str(e))
            break

if __name__ == "__main__":
    logger.info("Starting WebSocket server on port 5003...")
    server = pywsgi.WSGIServer(('', 5003), app, handler_class=WebSocketHandler)
    server.serve_forever()
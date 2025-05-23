from flask import Flask, jsonify
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return jsonify({"status": "healthy", "message": "Test server is running"})

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5001, debug=True)

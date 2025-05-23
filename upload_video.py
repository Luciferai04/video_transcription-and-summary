import os
import requests
import sys

def upload_video(file_path, server_url="http://localhost:5000/upload"):
    """
    Upload a video file to the server's /upload endpoint.

    Args:
        file_path (str): Path to the video file to upload.
        server_url (str): URL of the upload endpoint.

    Returns:
        tuple:
            - success (bool): True if upload is successful, False otherwise.
            - message (str): Server response or error message.
            - session_id (str or None): Session ID from the server if successful.
    """
    if not os.path.isfile(file_path):
        return False, f"File not found: {file_path}", None

    try:
        with open(file_path, "rb") as f:
            files = {'file': (os.path.basename(file_path), f)}
            response = requests.post(server_url, files=files)
        
        if response.status_code == 200:
            data = response.json()
            session_id = data.get("session_id")
            message = data.get("message", "Upload successful.")
            return True, message, session_id
        else:
            return False, f"Upload failed: HTTP {response.status_code} - {response.text}", None
    except requests.exceptions.RequestException as e:
        return False, f"Request error: {str(e)}", None
    except Exception as e:
        return False, f"Unexpected error: {str(e)}", None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upload_video.py <path_to_video_file> [server_url]")
        sys.exit(1)

    video_file = sys.argv[1]
    url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:5000/upload"

    success, msg, session = upload_video(video_file, url)
    if success:
        print(f"Upload successful! Session ID: {session}")
        print(f"You can use this session ID for further operations on the server.")
    else:
        print(f"Upload failed. Message: {msg}")


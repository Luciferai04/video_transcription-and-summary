import os
import sys
import logging
from gevent import monkey
monkey.patch_all()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from app import app, socketio
    logger.info("Successfully imported app and socketio")
except Exception as e:
    logger.error(f"Failed to import app: {str(e)}")
    sys.exit(1)

def create_required_directories():
    """Create necessary directories if they don't exist"""
    dirs = ['uploads', 'downloads']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_name}")

if __name__ == "__main__":
    try:
        # Create necessary directories
        create_required_directories()
        
        # Get host and port from environment variables or use defaults
        host = os.environ.get('HOST', '127.0.0.1')
        port = int(os.environ.get('PORT', 5001))
        
        # Start the application
        logger.info(f"Starting application on {host}:{port}")
        socketio.run(app, host=host, port=port)
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        sys.exit(1)

# For production servers (Gunicorn)
create_required_directories()
application = app

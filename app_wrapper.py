import webview
import threading
import sys
import os
from app import create_app

# Ensure the KMP_DUPLICATE_LIB_OK environment variable is set
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def start_flask():
    """Start the Flask application in a separate thread."""
    app = create_app()
    # Use 127.0.0.1 instead of localhost to avoid potential DNS resolution issues
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Give Flask a moment to start
    import time
    time.sleep(1)
    
    # Create the webview window
    webview.create_window("Memory Vault", "http://127.0.0.1:5000", width=1000, height=800)
    
    # Start the webview application
    webview.start(debug=True)
    
    # Clean exit when webview is closed
    sys.exit()
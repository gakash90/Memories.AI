import os
import tempfile
from datetime import datetime

import whisper
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder='static')

# Initialize Whisper model
model = whisper.load_model("tiny.en")

# Ensure upload directory exists
UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"success": False, "error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400

    # Create a temporary file with proper cleanup
    try:
        # Create a named temporary file that auto-deletes when closed
        with tempfile.NamedTemporaryFile(suffix='.wav', dir=UPLOAD_DIR, delete=False) as temp_file:
            temp_path = temp_file.name
            audio_file.save(temp_path)
        
        # Ensure the file is closed before processing
        try:
            # Transcribe the audio
            result = model.transcribe(temp_path)
            text = result["text"]

            return jsonify({
                "success": True,
                "transcription": text,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        finally:
            # Manually delete the file after processing
            try:
                os.unlink(temp_path)
            except PermissionError:
                # If deletion fails, try again after a small delay
                import time
                time.sleep(0.1)
                os.unlink(temp_path)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
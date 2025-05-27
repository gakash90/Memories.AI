"""
Blueprint for handling Whisper model download with database status tracking.
Modified implementation to use whisper.load_model() approach with database integration.
"""
from flask import Blueprint, jsonify, request
import os
import re
import threading
import time
import traceback
import whisper  # Import the whisper library directly

# Import the database functions
from db_schema import get_model_status, set_model_status

model_download_bp = Blueprint('model_download', __name__)

# Model information
MODEL_NAME = "whisper_turbo"  # Changed from turbo to base
MODEL_VERSION = "1.0"
# MODEL_DIR is now dynamic, generated in get_model_path()
# Known size of the medium model (approximate)
EXPECTED_MODEL_SIZE = 1510000000  

# Global variable to store the current progress
current_progress = {
    "percentage": 0,
    "downloaded": "",
    "total": "",
    "speed": "",
    "status": "Not started"
}

def get_model_path():
    """Get the path to the model directory"""
    # Use os.path.expanduser to get the user's home directory in a cross-platform way
    home_dir = os.path.expanduser('~')
    return os.path.join(home_dir, '.cache', 'whisper')

def check_model_files_exist():
    """
    Fast check to see if the required model files exist.
    Specifically looks for large-v3-turbo.pt in the whisper cache directory.
    """
    try:
        # Define the expected file path
        # Use os.path.expanduser to get the user's home directory in a cross-platform way
        home_dir = os.path.expanduser('~')
        whisper_cache_dir = os.path.join(home_dir, '.cache', 'whisper')
        model_file_path = os.path.join(whisper_cache_dir, 'large-v3-turbo.pt')  # base model file
        
        # Check for other possible paths where the file might be
        possible_paths = [
            model_file_path,
            os.path.join(whisper_cache_dir, 'large-v3-turbo.pt'),
            os.path.join(whisper_cache_dir, 'models', 'large-v3-turbo.pt')
        ]
        
        actual_file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                actual_file_path = path
                break
        
        # Check database status
        db_status = get_model_status(MODEL_NAME)
        
        # If DB says the model is downloaded but the file doesn't exist,
        # update the DB to mark it as not downloaded
        if db_status['is_downloaded'] and not actual_file_path:
            print(f"DB says model is downloaded but file doesn't exist. Updating DB.")
            set_model_status(MODEL_NAME, False, MODEL_VERSION, None)
            return False
        
        # If file exists
        if actual_file_path:
            # Verify it's a substantial file (at least 1MB)
            # For directories, check if they contain substantial files
            file_size = 0
            if os.path.isfile(actual_file_path):
                file_size = os.path.getsize(actual_file_path)
            elif os.path.isdir(actual_file_path):
                # If it's a directory, sum up the file sizes
                for root, dirs, files in os.walk(actual_file_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            file_size += os.path.getsize(file_path)
            
            if file_size < 1000000:  # 1MB minimum
                print(f"Model file {actual_file_path} exists but is too small ({file_size} bytes).")
                set_model_status(MODEL_NAME, False, MODEL_VERSION, None)
                return False
                
            # File exists and is substantial - update DB if needed
            if not db_status['is_downloaded']:
                print(f"Model file {actual_file_path} exists but DB says not downloaded. Updating DB.")
                set_model_status(MODEL_NAME, True, MODEL_VERSION, whisper_cache_dir)
            
            print(f"Model file found at {actual_file_path} with size {file_size} bytes")
            return True
        
        # File doesn't exist
        print("No model file found in any expected location")
        return False
            
    except Exception as e:
        print(f"Error checking model files: {str(e)}")
        traceback.print_exc()
        return False

# This function is no longer needed as we're using file monitoring instead of parsing output
# Kept as a stub for compatibility with existing code
def parse_progress(line):
    """Parse the progress line from whisper download output"""
    # No longer used - whisper.load_model() doesn't provide parseable output
    print(f"Log line (not parsed): {line}")
    return None

# Helper function to format file sizes
def format_size(size_bytes):
    if size_bytes is None:
        return "0B"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0 or unit == 'GB':
            return f"{size_bytes:.2f}{unit}"
        size_bytes /= 1024.0
    return "0B"

# Function to monitor download progress by checking file size
def monitor_download_progress():
    """Monitor the download progress by checking the file size periodically"""
    global current_progress
    
    # Expected model file path
    home_dir = os.path.expanduser('~')
    whisper_cache_dir = os.path.join(home_dir, '.cache', 'whisper')
    model_file_path = os.path.join(whisper_cache_dir, 'large-v3-turbo.pt')
    temp_file_path = os.path.join(whisper_cache_dir, 'large-v3-turbo.pt.tmp')
    part_file_path = os.path.join(whisper_cache_dir, 'large-v3-turbo.pt.part')

    # Use the known size of the medium model from the constant
    expected_total_size = EXPECTED_MODEL_SIZE
    
    start_time = time.time()
    last_size = 0
    last_check_time = start_time
    
    # Keep checking while download is in progress
    while current_progress["status"] == "Downloading" or current_progress["status"] == "Starting":
        try:
            # Check for temporary download files first
            if os.path.exists(part_file_path):
                file_path = part_file_path
            elif os.path.exists(temp_file_path):
                file_path = temp_file_path
            elif os.path.exists(model_file_path):
                file_path = model_file_path
            else:
                # No file found yet, might be preparing download
                time.sleep(0.5)
                continue
                
            # Get current file size
            current_size = os.path.getsize(file_path)
            current_time = time.time()
            time_diff = current_time - last_check_time
            
            # Calculate download speed
            if time_diff > 0:
                speed = (current_size - last_size) / time_diff
                speed_str = f"{format_size(speed)}/s"
            else:
                speed_str = "Calculating..."
            
            # Calculate percentage
            if current_size >= expected_total_size:
                percentage = 100
            else:
                percentage = int((current_size / expected_total_size) * 100)
                
            # Ensure percentage is within bounds
            percentage = max(0, min(99, percentage))  # Cap at 99% until verified complete
            
            # Update progress information
            current_progress.update({
                "percentage": percentage,
                "downloaded": format_size(current_size),
                "total": format_size(expected_total_size),
                "speed": speed_str,
                "status": "Downloading"
            })
            
            # Print progress
            print(f"Download progress: {percentage}% ({format_size(current_size)}/{format_size(expected_total_size)}) at {speed_str}")
            
            # Update for next iteration
            last_size = current_size
            last_check_time = current_time
            
            # If file exists and seems complete
            if os.path.exists(model_file_path) and current_size >= expected_total_size * 0.99:
                # Wait a moment to ensure file is fully written
                time.sleep(1)
                if os.path.exists(model_file_path):
                    current_progress.update({
                        "percentage": 100,
                        "downloaded": format_size(current_size),
                        "total": format_size(current_size),  # Use actual size as total
                        "speed": "Completed",
                        "status": "Completed"
                    })
                    print("Download appears to be complete.")
                    break
            
            # Sleep before next check
            time.sleep(1)
            
        except Exception as e:
            print(f"Error while monitoring download: {str(e)}")
            time.sleep(2)  # Wait longer if there's an error
            
    print("Download monitoring complete.")

def run_whisper_download():
    """Download the whisper model using the whisper library directly"""
    global current_progress
    
    try:
        current_progress["status"] = "Starting"
        
        # The whisper.load_model() doesn't accept a progress callback directly
        # We'll need to monitor the download indirectly
        print("Starting whisper model download...")
        
        # Set initial progress
        current_progress.update({
            "percentage": 0,
            "downloaded": "0MB",
            "total": "Unknown",
            "speed": "Unknown",
            "status": "Downloading"
        })
        
        # Start monitoring file size in the background
        monitor_thread = threading.Thread(target=monitor_download_progress)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Load the model, which will download it if not present
        # This will automatically download to the cache directory

        model = whisper.load_model("turbo", download_root=get_model_path())     

        # If we get here, download was successful
        current_progress["status"] = "Completed"
        current_progress["percentage"] = 100
        
        # Verify model exists
        if check_model_files_exist():
            # Update database when completed
            set_model_status(MODEL_NAME, True, MODEL_VERSION, get_model_path())
            print("Model download completed successfully!")
        else:
            current_progress["status"] = "Error: Process completed but model file not found"
            set_model_status(MODEL_NAME, False, MODEL_VERSION, None)
            print("Model download completed but file not found!")
            
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error downloading whisper model: {e}")
        print(error_traceback)
        current_progress["status"] = f"Error: {str(e)}"
        # Mark as failed in database
        set_model_status(MODEL_NAME, False, MODEL_VERSION, None)

@model_download_bp.route('/model/status', methods=['GET'])
def get_model_status_route():
    """Check the current status of the Whisper model"""
    global current_progress
    
    # First, check if the model file actually exists
    model_files_exist = check_model_files_exist()
    
    # If we're currently downloading, return the live progress
    if current_progress["status"] == "Downloading" or current_progress["status"] == "Starting":
        # Format response to match frontend expectations
        return jsonify({
            'status': current_progress["status"].lower(),
            'progress': current_progress["percentage"] / 100,
            'details': f"{current_progress['percentage']}% at {current_progress['speed']}"
        })
    
    # Check if completed or error
    if current_progress["status"] == "Completed":
        # Even if status says completed, verify the file exists
        if model_files_exist:
            return jsonify({
                'status': 'ready',
                'message': 'Whisper model is ready to use'
            })
        else:
            # File doesn't exist despite completed status
            # Reset progress and indicate download needed
            current_progress["status"] = "Not started"
            current_progress["percentage"] = 0
            return jsonify({
                'status': 'not_started',
                'progress': 0.0,
                'details': 'Model file missing, needs to be downloaded'
            })
    
    if current_progress["status"].startswith("Error"):
        return jsonify({
            'status': 'error',
            'error': current_progress["status"]
        })
    
    # Check if files exist
    if model_files_exist:
        return jsonify({
            'status': 'ready',
            'message': 'Whisper model is ready to use'
        })
    
    # Model truly isn't downloaded
    return jsonify({
        'status': 'not_started',
        'progress': 0.0,
        'details': 'Model needs to be downloaded'
    })

@model_download_bp.route('/model/download', methods=['POST'])
def start_model_download():
    """Start downloading the Whisper model"""
    global current_progress
    
    # Check if already downloading
    if current_progress["status"] == "Downloading" or current_progress["status"] == "Starting":
        return jsonify({
            'success': True,
            'message': 'Download already in progress'
        })
    
    # Check if already completed
    if current_progress["status"] == "Completed" or check_model_files_exist():
        return jsonify({
            'success': True,
            'message': 'Model already downloaded'
        })
    
    try:
        # Reset progress
        current_progress = {
            "percentage": 0,
            "downloaded": "",
            "total": "",
            "speed": "",
            "status": "Starting"
        }
        
        # Start the download in a separate thread
        thread = threading.Thread(target=run_whisper_download)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Model download started'
        })
    except Exception as e:
        error_msg = str(e)
        return jsonify({
            'success': False,
            'error': f"Failed to start download: {error_msg}"
        }), 500

@model_download_bp.route('/model/progress', methods=['GET'])
def get_download_progress():
    """Get the current progress of the model download"""
    global current_progress
    
    # First, check if the model file actually exists
    model_files_exist = check_model_files_exist()
    
    # Format the response to match frontend expectations
    if current_progress["status"] == "Downloading" or current_progress["status"] == "Starting":
        return jsonify({
            'status': current_progress["status"].lower(),
            'progress': current_progress["percentage"] / 100,
            'percentage': current_progress["percentage"],
            'downloaded': current_progress["downloaded"],
            'total': current_progress["total"],
            'speed': current_progress["speed"],
            'details': f"{current_progress['percentage']}% at {current_progress['speed']}"
        })
    
    if current_progress["status"] == "Completed":
        # Verify the file exists even if status says completed
        if model_files_exist:
            return jsonify({
                'status': 'completed',
                'progress': 1.0,
                'percentage': 100,
                'details': 'Model already downloaded'
            })
        else:
            # File doesn't exist despite completed status
            # Reset progress and indicate download needed
            current_progress["status"] = "Not started"
            current_progress["percentage"] = 0
            return jsonify({
                'status': 'not_started',
                'progress': 0.0,
                'percentage': 0,
                'details': 'Model file missing, needs to be downloaded'
            })
    
    if current_progress["status"].startswith("Error"):
        return jsonify({
            'status': 'error',
            'progress': current_progress["percentage"] / 100,
            'percentage': current_progress["percentage"],
            'details': current_progress["status"],
            'error': current_progress["status"]
        })
    
    # Check database status if not currently downloading
    if model_files_exist:
        return jsonify({
            'status': 'completed',
            'progress': 1.0,
            'percentage': 100,
            'details': 'Model already downloaded'
        })
    
    # No active download and file doesn't exist
    return jsonify({
        'status': 'not_started',
        'progress': 0.0,
        'percentage': 0,
        'details': 'Model needs to be downloaded'
    })
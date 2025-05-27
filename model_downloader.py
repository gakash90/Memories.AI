from flask import Blueprint, jsonify, request
import threading
import time
import os
import requests
import json
import logging
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create blueprint
model_downloader_bp = Blueprint('model_downloader', __name__)

# Global variables to track download status
download_status = {
    "status": "idle",  # idle, downloading, completed, error
    "completed": False,
    "error": None,
    "progress": 0,      # Download progress percentage
    "total_size": 0,    # Total size in MB
    "downloaded": 0,    # Downloaded size in MB
    "speed": 0,         # Download speed in MB/s
    "eta": "--"         # Estimated time remaining
}

def check_model_presence(model_name="gemma3"):
    """Check if the model exists using ollama ls command."""
    try:
        # Execute ollama ls command
        result = subprocess.run(
            ["ollama", "ls"],
            capture_output=True,
            text=True,
            check=True,
        )
        
        # Get the output and split into lines
        lines = result.stdout.splitlines()
        
        # Skip header and check each line
        for line in lines[1:]:  # Skip "NAME ID SIZE MODIFIED" header
            if line.strip():  # Ignore empty lines
                name = line.split()[0]  # First column is NAME
                if name == model_name or name == f"{model_name}:latest":
                    logger.info(f"Model {model_name} found in ollama ls output")
                    return True
        logger.info(f"Model {model_name} not found in ollama ls output")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing ollama ls: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("Ollama CLI not found. Is Ollama installed?")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking model presence: {str(e)}")
        return False

def download_model_task():
    """Background task to download the model through Ollama API"""
    global download_status
    
    try:
        # Update status to downloading
        download_status["status"] = "downloading"
        download_status["completed"] = False
        download_status["error"] = None
        download_status["progress"] = 0
        
        logger.info("Testing connection to Ollama")
        # Check Ollama connection first
        try:
            test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if test_response.status_code != 200:
                logger.error(f"Ollama check failed: {test_response.status_code} - {test_response.text}")
                download_status["status"] = "error"
                download_status["error"] = f"Cannot connect to Ollama: HTTP {test_response.status_code}"
                return
            logger.info("Ollama connection successful")
        except requests.RequestException as e:
            logger.error(f"Ollama connection error: {str(e)}")
            download_status["status"] = "error" 
            download_status["error"] = "Cannot connect to Ollama server. Is it running?"
            return
        
        # Check if model already exists using ollama ls
        if check_model_presence():
            logger.info("Model already exists in Ollama")
            download_status["status"] = "completed"
            download_status["completed"] = True
            download_status["progress"] = 100
            return
        
        # Start the model download
        logger.info("Starting model download via Ollama")
        
        # Call Ollama API to pull the model
        response = requests.post(
            "http://localhost:11434/api/pull",
            json={"name": "gemma3"},
            stream=True
        )
        
        if response.status_code != 200:
            download_status["status"] = "error"
            download_status["error"] = f"Error starting download: {response.text}"
            return
        
        # Variables to track download progress
        start_time = time.time()
        last_update_time = start_time
        last_downloaded = 0
        
        # Process the streaming response
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    
                    # Check for success status
                    if 'status' in data and data['status'] == 'success':
                        download_status["status"] = "completed"
                        download_status["completed"] = True
                        download_status["progress"] = 100
                        logger.info("Model download completed successfully")
                        break
                    
                    # Update progress information if available
                    current_time = time.time()
                    if 'total' in data and 'completed' in data:
                        # Update sizes (convert to MB)
                        total_mb = data['total'] / (1024 * 1024)
                        downloaded_mb = data['completed'] / (1024 * 1024)
                        
                        # Calculate progress percentage
                        if data['total'] > 0:
                            progress = (data['completed'] / data['total']) * 100
                        else:
                            progress = 0
                            
                        # Calculate download speed (MB/s) - update every second
                        if current_time - last_update_time >= 1.0:
                            elapsed = current_time - last_update_time
                            downloaded_since_last = downloaded_mb - last_downloaded
                            speed = downloaded_since_last / elapsed if elapsed > 0 else 0
                            
                            # Calculate ETA
                            remaining_mb = total_mb - downloaded_mb
                            if speed > 0:
                                seconds_remaining = remaining_mb / speed
                                # Format ETA
                                if seconds_remaining < 60:
                                    eta = f"{int(seconds_remaining)}s"
                                else:
                                    minutes = int(seconds_remaining // 60)
                                    seconds = int(seconds_remaining % 60)
                                    eta = f"{minutes}m{seconds}s"
                            else:
                                eta = "--"
                                
                            # Update status
                            download_status["speed"] = round(speed, 1)
                            download_status["eta"] = eta
                            
                            # Update for next calculation
                            last_update_time = current_time
                            last_downloaded = downloaded_mb
                        
                        # Update status
                        download_status["progress"] = round(progress, 1)
                        download_status["total_size"] = round(total_mb, 1)
                        download_status["downloaded"] = round(downloaded_mb, 1)
                        
                        logger.debug(f"Progress: {progress:.1f}% - {downloaded_mb:.1f}MB/{total_mb:.1f}MB")
                        
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.error(f"Error processing status: {str(e)}")
    
    except requests.RequestException as e:
        download_status["status"] = "error"
        download_status["error"] = f"Connection error: {str(e)}"
        logger.error(f"Download connection error: {str(e)}")
    except Exception as e:
        download_status["status"] = "error"
        download_status["error"] = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected download error: {str(e)}")

@model_downloader_bp.route('/api/model/start-download', methods=['POST'])
def start_download():
    """Start downloading the model if not already downloading"""
    global download_status
    
    logger.info("Start download endpoint called")
    
    # Check if model already exists using ollama ls
    if check_model_presence():
        logger.info("Model already exists in Ollama")
        download_status["status"] = "completed"
        download_status["completed"] = True
        download_status["progress"] = 100
        return jsonify({'success': True, 'message': 'Model already available'})
    
    if download_status["status"] == "downloading":
        logger.info("Download already in progress")
        return jsonify({'success': False, 'message': 'Download already in progress'})
    
    if download_status["completed"]:
        logger.info("Model already downloaded")
        return jsonify({'success': True, 'message': 'Model already downloaded'})
    
    # Start download in background thread
    logger.info("Starting download thread")
    download_thread = threading.Thread(target=download_model_task)
    download_thread.daemon = True
    download_thread.start()
    
    return jsonify({'success': True, 'message': 'Download started'})

@model_downloader_bp.route('/api/model/status', methods=['GET'])
def get_status():
    """Get current download status"""
    global download_status
    
    # Double-check with ollama ls if we're not already in completed state
    if not download_status["completed"]:
        if check_model_presence():
            logger.info("Model found in Ollama during status check")
            download_status["status"] = "completed"
            download_status["completed"] = True
            download_status["progress"] = 100
            
    return jsonify(download_status)

# Function to check if model is downloaded
def is_model_downloaded():
    """Check if the model has been downloaded completely"""
    # Call the status endpoint to ensure we have the latest info
    get_status()
    return download_status["completed"]
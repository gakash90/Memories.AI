from flask import Blueprint, request, jsonify, session, send_file
import os
import datetime
import time
import shutil
import uuid
import tempfile
import librosa
import numpy as np
import json
from auth_blueprint import login_required
import torchaudio
import whisper
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy import signal
import webrtcvad
import io

upload_bp = Blueprint('upload', __name__)

# Initialize variables to store the models once loaded
_whisper_model = None
_voice_encoder = None
_vad = None

# Configurable parameters for diarization
DIARIZATION_CONFIG = {
    "min_segment_length": 0.5,      # Minimum segment length in seconds
    "word_gap_threshold": 0.3,       # Gap between words to create a new segment (seconds)
    "overlap_window": 0.5,           # Overlap between segments for embedding extraction
    "embedding_frame_length": 2.0,   # Length of audio frame for embedding extraction
    "clustering_method": "average",  # Linkage method for clustering: 'ward', 'complete', 'average'
    "distance_threshold": 0.3,       # Distance threshold for clustering (used if num_speakers not provided)
    "smooth_window": 3,              # Window size for median filtering of speaker labels
    "whisper_model_size": "base", # Options: "tiny", "base", "small", "medium", "large-v3", "turbo"
    "use_pca": True,                 # Whether to use PCA for embedding dimensionality reduction
    "pca_components": 32,            # Number of PCA components to keep
    "vad_aggressiveness": 3,         # WebRTC VAD aggressiveness (0-3)
    "vad_frame_ms": 30,              # VAD frame size in milliseconds (10, 20, or 30)
    "min_vad_speech_duration": 0.2,  # Minimum speech duration to keep a segment after VAD
}

def get_vad():
    """Lazy load the VAD model"""
    global _vad
    if _vad is None:
        print("Initializing WebRTC VAD...")
        _vad = webrtcvad.Vad(DIARIZATION_CONFIG["vad_aggressiveness"])
    return _vad

def convert_webm_to_mp3(webm_file_path):
    """Convert WebM file to WAV format using pydub which can handle WebM with Opus codec
    
    Args:
        webm_file_path: Path to the WebM file
        
    Returns:
        tuple: (Path to the converted file, success flag)
    """
    try:
        print(f"Converting WebM file to WAV: {webm_file_path}")
        
        # We'll convert to WAV instead of MP3 since WAV is well-supported by Whisper
        wav_file_path = webm_file_path.rsplit('.', 1)[0] + '.wav'
        
        # Use pydub for conversion
        from pydub import AudioSegment
        audio = AudioSegment.from_file(webm_file_path, format="webm")
        audio.export(wav_file_path, format="wav")
        
        print(f"Successfully converted WebM to WAV: {wav_file_path}")
        return wav_file_path, True
    except Exception as e:
        print(f"Error converting WebM to WAV: {str(e)}")
        
        # Fallback to direct loading with librosa
        try:
            print("Attempting fallback conversion with librosa...")
            import librosa
            import soundfile as sf
            
            # Load with librosa which can sometimes handle WebM
            y, sr = librosa.load(webm_file_path, sr=16000, mono=True)
            
            # Save as WAV file
            wav_file_path = webm_file_path.rsplit('.', 1)[0] + '.wav'
            sf.write(wav_file_path, y, sr, format='WAV')
            
            print(f"Fallback conversion successful: {wav_file_path}")
            return wav_file_path, True
        except Exception as e2:
            print(f"Fallback conversion failed: {str(e2)}")
            return None, False

def preprocess_audio(file_path):
    """Preprocess audio using torchaudio to bypass FFmpeg dependency"""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze().numpy().astype("float32")

def get_diarization_models():
    """Lazy load the diarization models only when needed"""
    global _whisper_model, _voice_encoder
    if _whisper_model is None:
        print(f"Loading Whisper model ({DIARIZATION_CONFIG['whisper_model_size']}) for diarization...")
        _whisper_model = whisper.load_model(DIARIZATION_CONFIG["whisper_model_size"])
    
    if _voice_encoder is None:
        print("Loading voice encoder model...")
        _voice_encoder = VoiceEncoder()
    
    return _whisper_model, _voice_encoder

def transcribe_audio_with_timestamps(audio_path):
    """Transcribe audio using Whisper with timestamps"""
    print(f"Transcribing audio with timestamps from {audio_path}")
    whisper_model, _ = get_diarization_models()
    
    # Preprocess audio using torchaudio instead of relying on Whisper's built-in processing
    audio = preprocess_audio(audio_path)
    
    # Use the preprocessed audio array instead of the file path
    result = whisper_model.transcribe(audio, word_timestamps=True)
    return result

def apply_vad(audio, sample_rate=16000):
    """Apply Voice Activity Detection to identify speech segments"""
    vad = get_vad()
    
    # Frame parameters based on the VAD configuration
    frame_ms = DIARIZATION_CONFIG["vad_frame_ms"]
    frame_len = int(sample_rate * (frame_ms / 1000.0))
    
    # Pad audio to ensure we have complete frames
    pad_size = frame_len - (len(audio) % frame_len)
    if pad_size < frame_len:
        audio = np.pad(audio, (0, pad_size), 'constant')
    
    # Convert float audio to int16 for VAD
    audio_int16 = (audio * 32767).astype(np.int16).tobytes()
    
    # Process frames
    speech_frames = []
    for i in range(0, len(audio) - frame_len + 1, frame_len):
        frame = audio_int16[i*2:(i+frame_len)*2]  # *2 because each int16 is 2 bytes
        if vad.is_speech(frame, sample_rate):
            speech_frames.append(1)
        else:
            speech_frames.append(0)
    
    # Convert to numpy array for easier manipulation
    speech_mask = np.array(speech_frames)
    
    # Smooth the speech mask
    speech_mask = signal.medfilt(speech_mask, 5)
    
    return speech_mask

def segment_audio(audio_path, min_segment_length=None):
    """Segment audio based on voice activity and silence"""
    if min_segment_length is None:
        min_segment_length = DIARIZATION_CONFIG["min_segment_length"]
    
    print("Segmenting audio based on speech activity")
    
    # Load audio file using librosa
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Apply VAD to get speech mask
    speech_mask = apply_vad(y, sr)
    
    # Get word-level transcription to use as segments
    result = transcribe_audio_with_timestamps(audio_path)
    
    segments = []
    current_segment = {"start": None, "end": None, "text": "", "words": []}
    
    word_gap_threshold = DIARIZATION_CONFIG["word_gap_threshold"]
    
    for segment in result["segments"]:
        for word in segment.get("words", []):
            # If this is the first word or if the gap is too large
            if current_segment["start"] is None:
                current_segment["start"] = word["start"]
                current_segment["text"] = word["word"]
                current_segment["words"].append(word)
            elif word["start"] - current_segment["end"] > word_gap_threshold:  # Gap means new segment
                current_segment["end"] = current_segment["words"][-1]["end"]
                segments.append(current_segment)
                current_segment = {"start": word["start"], "end": None, "text": word["word"], "words": [word]}
            else:
                current_segment["text"] += word["word"]
                current_segment["words"].append(word)
            
            current_segment["end"] = word["end"]
    
    # Add the last segment if it exists
    if current_segment["start"] is not None:
        current_segment["end"] = current_segment["words"][-1]["end"]
        segments.append(current_segment)
    
    # Extract audio for each segment with overlap for better speaker identification
    audio_segments = []
    
    # Add some padding for each segment for better speaker identification
    overlap_window = DIARIZATION_CONFIG["overlap_window"]
    
    for i, segment in enumerate(segments):
        start_time = max(0, segment["start"] - overlap_window/2)
        end_time = min(len(y)/sr, segment["end"] + overlap_window/2)
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Ensure we don't go out of bounds
        if start_sample >= len(y) or end_sample > len(y) or start_sample >= end_sample:
            continue
            
        segment_audio = y[start_sample:end_sample]
        
        # Skip very short segments
        if len(segment_audio) / sr < min_segment_length:
            continue
        
        # Check if the segment has enough speech based on VAD
        # Convert time to frame indices
        start_frame = int(start_time / (DIARIZATION_CONFIG["vad_frame_ms"] / 1000.0))
        end_frame = int(end_time / (DIARIZATION_CONFIG["vad_frame_ms"] / 1000.0))
        
        # Ensure indices are within bounds
        start_frame = max(0, min(start_frame, len(speech_mask) - 1))
        end_frame = max(0, min(end_frame, len(speech_mask) - 1))
        
        if end_frame > start_frame:
            segment_speech_ratio = np.mean(speech_mask[start_frame:end_frame])
            # If the segment doesn't have enough speech, skip it
            if segment_speech_ratio < 0.3:  # At least 30% of the segment should be speech
                continue
        
        audio_segments.append({
            "audio": segment_audio,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })
    
    return audio_segments, result["text"]

def extract_embeddings_with_sliding_window(audio_data, sr=16000):
    """Extract speaker embeddings using sliding windows for better representation"""
    _, voice_encoder = get_diarization_models()
    
    # If the audio is too short, return a single embedding
    if len(audio_data) / sr < DIARIZATION_CONFIG["embedding_frame_length"]:
        processed_wav = preprocess_wav(audio_data, source_sr=sr)
        return voice_encoder.embed_utterance(processed_wav)
    
    # Use sliding windows for longer segments
    frame_length = int(DIARIZATION_CONFIG["embedding_frame_length"] * sr)
    hop_length = frame_length // 2  # 50% overlap
    
    embeddings = []
    
    for i in range(0, len(audio_data) - frame_length + 1, hop_length):
        frame = audio_data[i:i+frame_length]
        processed_frame = preprocess_wav(frame, source_sr=sr)
        embedding = voice_encoder.embed_utterance(processed_frame)
        embeddings.append(embedding)
    
    # Return average embedding
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        # Fallback to using the entire segment
        processed_wav = preprocess_wav(audio_data, source_sr=sr)
        return voice_encoder.embed_utterance(processed_wav)

def get_speaker_embeddings(audio_segments):
    """Extract speaker embeddings from audio segments"""
    print("Extracting speaker embeddings")
    embeddings = []
    
    for segment in audio_segments:
        if len(segment["audio"]) == 0:
            continue
            
        # Get embedding using sliding window approach
        embedding = extract_embeddings_with_sliding_window(segment["audio"])
        embeddings.append(embedding)
    
    return np.array(embeddings)

def estimate_num_speakers(embeddings):
    """Estimate the optimal number of speakers using silhouette score"""
    min_speakers = 1
    max_speakers = min(8, len(embeddings) // 3)  # More reasonable upper bound
    
    if len(embeddings) < 4:  # Too few segments to estimate reliably
        return min(2, len(embeddings))
    
    best_score = -1
    best_n = 2
    
    # Try different numbers of speakers and keep the one with the best silhouette score
    for n in range(min_speakers, max_speakers + 1):
        if len(embeddings) <= n:  # Skip if we have fewer embeddings than clusters
            continue
            
        clustering = AgglomerativeClustering(n_clusters=n)
        labels = clustering.fit_predict(embeddings)
        
        # Skip if we have clusters with only one sample
        unique_labels, counts = np.unique(labels, return_counts=True)
        if min(counts) < 2:
            continue
            
        score = silhouette_score(embeddings, labels)
        
        if score > best_score:
            best_score = score
            best_n = n
    
    print(f"Estimated number of speakers: {best_n}")
    return best_n

def cluster_speakers(embeddings, num_speakers=None):
    """Cluster speaker embeddings to identify unique speakers"""
    print("Clustering speakers")
    
    # Apply PCA to reduce dimensionality if configured
    if DIARIZATION_CONFIG["use_pca"] and len(embeddings) > DIARIZATION_CONFIG["pca_components"]:
        pca = PCA(n_components=min(DIARIZATION_CONFIG["pca_components"], len(embeddings)-1))
        embeddings = pca.fit_transform(embeddings)
        print(f"Applied PCA: reduced dimensions to {embeddings.shape[1]}")
    
    # If no number of speakers is provided, estimate it
    if num_speakers is None:
        num_speakers = estimate_num_speakers(embeddings)
    
    print(f"Using {num_speakers} speakers for clustering")
    
    # Perform clustering with the specified linkage method
    clustering = AgglomerativeClustering(
        n_clusters=num_speakers,
        linkage=DIARIZATION_CONFIG["clustering_method"]
    )
    
    labels = clustering.fit_predict(embeddings)
    
    return labels

def smooth_speaker_labels(labels, window_size=None):
    """Apply median filtering to smooth speaker transitions"""
    if window_size is None:
        window_size = DIARIZATION_CONFIG["smooth_window"]
        
    if len(labels) <= window_size:
        return labels
        
    smoothed = signal.medfilt(labels, window_size)
    return smoothed

def process_audio_file_with_diarization(file_path, num_speakers=None):
    """Process audio file to transcribe and identify speakers"""
    print(f"Processing audio file with diarization: {file_path}")
    
    # Segment the audio
    audio_segments, full_transcript = segment_audio(file_path)
    
    # Get speaker embeddings
    embeddings = get_speaker_embeddings(audio_segments)
    
    # Skip speaker identification if we couldn't extract embeddings
    if len(embeddings) == 0:
        return {"error": "Could not extract speaker information from the audio", "full_transcript": full_transcript}
    
    # Cluster to identify speakers
    speaker_labels = cluster_speakers(embeddings, num_speakers)
    
    # Apply smoothing to speaker labels
    if len(speaker_labels) > 3:  # Only smooth if we have enough segments
        speaker_labels = smooth_speaker_labels(speaker_labels)
    
    # Assign speakers to segments
    result = []
    for i, (segment, speaker_id) in enumerate(zip(audio_segments, speaker_labels)):
        result.append({
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": segment["text"],
            "speaker": f"Speaker {speaker_id + 1}"
        })
    
    # Sort segments by start time
    result = sorted(result, key=lambda x: x["start"])
    
    # Post-process: Smooth isolated speaker segments
    # If a single segment is surrounded by the same speaker, change it to match
    if len(result) >= 3:
        for i in range(1, len(result) - 1):
            if (result[i-1]["speaker"] == result[i+1]["speaker"] and
                result[i]["speaker"] != result[i-1]["speaker"]):
                # Short isolated segment
                if result[i]["end"] - result[i]["start"] < 1.5:
                    result[i]["speaker"] = result[i-1]["speaker"]
    
    return {
        "segments": result,
        "full_transcript": full_transcript
    }

def adjust_diarization_config(config_updates=None):
    """Update diarization configuration with provided values"""
    global DIARIZATION_CONFIG
    if config_updates:
        for key, value in config_updates.items():
            if key in DIARIZATION_CONFIG:
                print(f"Updating diarization config: {key} = {value}")
                DIARIZATION_CONFIG[key] = value
    return DIARIZATION_CONFIG

@upload_bp.route('/upload', methods=['POST'])
@login_required
def upload_audio():
    try:
        user_id = session['user_id']
        
        from utils import get_user_dirs, encrypt_file_in_place, decrypt_file, save_transcription, get_audio_duration
        
        user_dirs = get_user_dirs(user_id)
        
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"success": False, "error": "No selected file"}), 400

        original_filename = audio_file.filename
        print(f"Received file: {original_filename}, Content type: {audio_file.content_type}")

        if '.' in original_filename:
            file_ext = original_filename.rsplit('.', 1)[1].lower()
        else:
            content_type = audio_file.content_type.lower()
            if 'webm' in content_type:
                file_ext = 'webm'
            elif 'ogg' in content_type:
                file_ext = 'ogg'
            elif 'mp3' in content_type or 'mpeg' in content_type:
                file_ext = 'mp3'
            elif 'wav' in content_type:
                file_ext = 'wav'
            else:
                file_ext = 'mp3'

        print(f"Determined file extension: {file_ext}")

        supported_formats = ['mp3', 'wav', 'webm', 'ogg']
        if file_ext.lower() not in supported_formats:
            return jsonify({"success": False, "error": f"Unsupported file type. Please upload {', '.join(supported_formats)} files."}), 400

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the audio file directly to the user's audio directory with original format
        audio_filename = f"audio_{timestamp}.{file_ext}"
        audio_path = os.path.join(user_dirs['audio'], audio_filename)
        
        audio_file.save(audio_path)
        print(f"Saved audio file to: {audio_path}")
        
        # Convert WebM to WAV if needed since Whisper doesn't support WebM well
        final_audio_path = audio_path
        final_audio_filename = audio_filename
        
        if file_ext.lower() == 'webm':
            print("WebM format detected, converting to WAV for compatibility with Whisper")
            
            # Make sure pydub is installed (add to requirements.txt)
            try:
                # First attempt using pydub
                from pydub import AudioSegment
                wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
                audio = AudioSegment.from_file(audio_path, format="webm")
                audio.export(wav_path, format="wav")
                
                # If successful, use the WAV file
                final_audio_path = wav_path
                final_audio_filename = os.path.basename(wav_path)
                print(f"Successfully converted WebM to WAV: {final_audio_path}")
                
                # Remove the original WebM file to save space
                try:
                    os.remove(audio_path)
                    print(f"Removed original WebM file: {audio_path}")
                except Exception as e:
                    print(f"Warning: Failed to remove original WebM file: {str(e)}")
                    
            except Exception as e:
                print(f"Error converting WebM using pydub: {str(e)}")
                
                # Fallback to librosa
                try:
                    print("Attempting fallback conversion with librosa...")
                    import librosa
                    import soundfile as sf
                    
                    # Load with librosa
                    y, sr = librosa.load(audio_path, sr=16000, mono=True)
                    
                    # Save as WAV file
                    wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
                    sf.write(wav_path, y, sr, format='WAV')
                    
                    # If successful, use the WAV file
                    final_audio_path = wav_path
                    final_audio_filename = os.path.basename(wav_path)
                    print(f"Fallback conversion successful: {final_audio_path}")
                    
                    # Remove the original WebM file
                    try:
                        os.remove(audio_path)
                        print(f"Removed original WebM file: {audio_path}")
                    except Exception as e:
                        print(f"Warning: Failed to remove original WebM file: {str(e)}")
                except Exception as e2:
                    print(f"All WebM conversion methods failed: {str(e2)}")
                    print("Continuing with original WebM file, but transcription may fail")
        
        # Get duration: prefer provided duration, fallback to get_audio_duration
        duration = 0
        if 'duration' in request.form:
            try:
                duration = float(request.form['duration'])
                if duration <= 0:
                    print("Provided duration is zero or negative, calculating duration")
                    duration = get_audio_duration(final_audio_path)
                else:
                    print(f"Using provided duration: {duration} seconds")
            except ValueError:
                print("Invalid duration provided, calculating duration")
                duration = get_audio_duration(final_audio_path)
        else:
            duration = get_audio_duration(final_audio_path)
        print(f"Final duration: {duration} seconds")
        
        # Check for diarization configuration parameters
        diarization_config_updates = {}
        for key in DIARIZATION_CONFIG.keys():
            if key in request.form:
                try:
                    # Handle different parameter types
                    if key in ["use_pca"]:
                        value = request.form[key].lower() in ["true", "1", "yes"]
                    elif key in ["whisper_model_size", "clustering_method"]:
                        value = request.form[key]
                    else:
                        value = float(request.form[key])
                    diarization_config_updates[key] = value
                except ValueError:
                    print(f"Invalid value for {key}: {request.form[key]}")
        
        # Apply any configuration updates
        if diarization_config_updates:
            adjust_diarization_config(diarization_config_updates)
        
        # Encrypt the audio file after duration calculation
        try:
            encrypt_file_in_place(final_audio_path, user_id)
            print(f"File encrypted successfully: {final_audio_path}")
        except Exception as e:
            print(f"Warning: File encryption failed: {str(e)}")

        temp_decrypted_path = os.path.join('temp', f"decrypted_{timestamp}{os.path.splitext(final_audio_path)[1]}")
        try:
            decrypt_file(final_audio_path, temp_decrypted_path, user_id)
            
            # Process audio with diarization
            try:
                print("Starting diarization processing...")
                # Get number of speakers if provided
                num_speakers = None
                if 'num_speakers' in request.form and request.form['num_speakers'].isdigit():
                    num_speakers = int(request.form['num_speakers'])
                        
                diarization_result = process_audio_file_with_diarization(temp_decrypted_path, num_speakers)
                print(f"Diarization completed with {len(diarization_result.get('segments', []))} segments")
                
                # Extract transcription from diarization result
                text = diarization_result.get("full_transcript", "")
                
                # Save transcription
                if "error" in diarization_result:
                    print(f"Diarization failed: {diarization_result['error']}")
                    # Still save the transcription if available
                    transcription_entry = save_transcription(user_id, text, final_audio_path, duration)
                else:
                    transcription_entry = save_transcription(
                        user_id, 
                        text, 
                        final_audio_path, 
                        duration, 
                        diarization_result
                    )
                
            except Exception as e:
                print(f"Error during audio processing: {str(e)}")
                # Fallback to basic transcription if diarization fails
                try:
                    result = transcribe_audio_with_timestamps(temp_decrypted_path)
                    text = result["text"]
                    transcription_entry = save_transcription(user_id, text, final_audio_path, duration)
                except Exception as e2:
                    print(f"Fallback transcription failed: {str(e2)}")
                    return jsonify({"success": False, "error": f"Audio processing failed: {str(e)}"}), 500

            if os.path.exists(temp_decrypted_path):
                os.remove(temp_decrypted_path)

            return jsonify({
                "success": True,
                "transcription": text,
                "all_transcriptions": [transcription_entry]
            })

        except Exception as e:
            print(f"Transcription error: {str(e)}")
            if os.path.exists(temp_decrypted_path):
                os.remove(temp_decrypted_path)
            return jsonify({"success": False, "error": f"Transcription failed: {str(e)}"}), 500

    except Exception as e:
        import traceback
        print(f"Upload error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

def ensure_temp_dirs_exist():
    temp_dir = os.path.join(os.getcwd(), 'temp')
    playback_dir = os.path.join(temp_dir, 'playback')
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    if not os.path.exists(playback_dir):
        os.makedirs(playback_dir)
        
    return temp_dir, playback_dir

def get_user_playback_dir(user_id):
    _, playback_dir = ensure_temp_dirs_exist()
    user_playback_dir = os.path.join(playback_dir, str(user_id))
    
    if not os.path.exists(user_playback_dir):
        os.makedirs(user_playback_dir)
        
    return user_playback_dir

def cleanup_old_playback_files(user_id, max_age=3600):
    user_playback_dir = get_user_playback_dir(user_id)
    current_time = time.time()
    
    try:
        for filename in os.listdir(user_playback_dir):
            file_path = os.path.join(user_playback_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age:
                    os.remove(file_path)
                    print(f"Removed old playback file: {file_path}")
    except Exception as e:
        print(f"Error cleaning up playback files: {str(e)}")

@upload_bp.route('/play_audio/<filename>', methods=['GET'])
@login_required
def play_audio_file(filename):
    try:
        current_user_id = session.get('user_id')
        if not current_user_id:
            return jsonify({"success": False, "error": "User not authenticated"}), 401
        
        from utils import get_user_dirs, decrypt_file
        
        user_dirs = get_user_dirs(current_user_id)
        encrypted_path = os.path.join(user_dirs['audio'], filename)
        
        if not os.path.exists(encrypted_path):
            return jsonify({"success": False, "error": "Audio file not found"}), 404
        
        playback_id = str(uuid.uuid4())
        user_playback_dir = get_user_playback_dir(current_user_id)
        cleanup_old_playback_files(current_user_id)
        
        temp_decrypted_filename = f"{playback_id}_{filename}"
        temp_decrypted_path = os.path.join(user_playback_dir, temp_decrypted_filename)
        
        try:
            decrypt_file(encrypted_path, temp_decrypted_path, current_user_id)
            print(f"File decrypted successfully to: {temp_decrypted_path}")
        except Exception as e:
            print(f"Error decrypting audio file: {str(e)}")
            return jsonify({"success": False, "error": "Failed to decrypt audio file"}), 500
        
        # Determine MIME type based on file extension
        if filename.endswith('.mp3'):
            mime_type = 'audio/mpeg'
        elif filename.endswith('.wav'):
            mime_type = 'audio/wav'
        elif filename.endswith('.ogg'):
            mime_type = 'audio/ogg'
        elif filename.endswith('.webm'):
            mime_type = 'audio/webm'
        else:
            mime_type = 'audio/mpeg'  # Default fallback
            
        abs_temp_path = os.path.abspath(temp_decrypted_path)
        print(f"Playing audio from complete file path: {abs_temp_path}")
        
        response = send_file(
            temp_decrypted_path,
            mimetype=mime_type,
            as_attachment=False,
            conditional=True
        )
        
        # Add cleanup function to remove the temporary file after streaming
        @response.call_on_close
        def remove_temp_file():
            try:
                if os.path.exists(temp_decrypted_path):
                    os.remove(temp_decrypted_path)
                    print(f"Removed temporary decrypted file: {temp_decrypted_path}")
            except Exception as e:
                print(f"Error removing temporary file: {str(e)}")
        
        return response
            
    except Exception as e:
        import traceback
        print(f"Error serving audio file {filename}: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

@upload_bp.route('/user_data/<user_id>/audio/<filename>', methods=['GET'])
@login_required
def serve_user_audio_file(user_id, filename):
    try:
        current_user_id = session.get('user_id')
        if str(current_user_id) != str(user_id):
            return jsonify({"success": False, "error": "Access denied"}), 403
        
        from utils import get_user_dirs, decrypt_file
        
        user_dirs = get_user_dirs(current_user_id)
        encrypted_path = os.path.join(user_dirs['audio'], filename)
        
        if not os.path.exists(encrypted_path):
            return jsonify({"success": False, "error": "File not found"}), 404
        
        user_playback_dir = get_user_playback_dir(current_user_id)
        playback_id = str(uuid.uuid4())
        temp_decrypted_filename = f"{playback_id}_{filename}"
        temp_decrypted_path = os.path.join(user_playback_dir, temp_decrypted_filename)
        
        try:
            decrypt_file(encrypted_path, temp_decrypted_path, current_user_id)
            
            # Determine the correct MIME type based on file extension
            if filename.endswith('.mp3'):
                mime_type = 'audio/mpeg'
            elif filename.endswith('.wav'):
                mime_type = 'audio/wav'
            elif filename.endswith('.ogg'):
                mime_type = 'audio/ogg'
            elif filename.endswith('.webm'):
                mime_type = 'audio/webm'
            else:
                mime_type = 'audio/mpeg'  # Default fallback
            
            response = send_file(
                temp_decrypted_path,
                mimetype=mime_type,
                as_attachment=False,
                conditional=True
            )
            
            @response.call_on_close
            def remove_temp_file():
                try:
                    if os.path.exists(temp_decrypted_path):
                        os.remove(temp_decrypted_path)
                except Exception as e:
                    print(f"Error removing temp file: {str(e)}")
            
            return response
            
        except Exception as e:
            print(f"Error decrypting/serving audio file: {str(e)}")
            if os.path.exists(temp_decrypted_path):
                os.remove(temp_decrypted_path)
            return jsonify({"success": False, "error": "Failed to process audio file"}), 500
            
    except Exception as e:
        print(f"Error serving audio file {filename}: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
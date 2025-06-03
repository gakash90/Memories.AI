# smart_audio_processor.py - Intelligent Whisper processing using WhisperX logic

import whisper
import librosa
import soundfile as sf
import numpy as np
import os
import logging
import gc
from collections import Counter

logger = logging.getLogger(__name__)

class SmartWhisperProcessor:
    """
    Intelligent Whisper processor using WhisperX-inspired logic
    Uses standard Whisper with smart language detection and processing
    """
    
    def __init__(self, whisper_model):
        self.whisper_model = whisper_model
        
        # Configuration optimized for your use case
        self.config = {
            "max_audio_duration": 30 * 60,  # 30 minutes max per chunk
            "sample_duration_for_detection": 60,  # 1 minute for language detection
            "quality_threshold": 0.6,  # Repetition threshold
            "min_transcript_length": 10,
        }
    
    def detect_language_from_sample(self, audio_path, sample_duration=None):
        """
        Detect language from audio sample using Whisper's built-in detection
        Inspired by your WhisperX implementation
        """
        if sample_duration is None:
            sample_duration = self.config["sample_duration_for_detection"]
            
        logger.info(f"Detecting language from audio sample: {audio_path}")
        
        try:
            # Load audio sample (first 60 seconds)
            audio = whisper.load_audio(audio_path)
            if len(audio) > sample_duration * 16000:  # 60 seconds at 16kHz
                audio = audio[:sample_duration * 16000]
            
            # Use Whisper's built-in language detection
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)
            
            # Detect language probabilities
            _, probs = self.whisper_model.detect_language(mel)
            
            # Get top languages
            top_languages = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            logger.info(f"Language probabilities: {top_languages}")
            
            # Analyze results (your WhisperX logic adapted)
            hindi_prob = probs.get('hi', 0.0)
            english_prob = probs.get('en', 0.0)
            
            # Decision logic for Hinglish vs English
            if hindi_prob > 0.3 and english_prob > 0.2:
                # Both languages present - likely Hinglish
                detected_language = 'hi'  # Use Hindi model for Hinglish
                confidence = hindi_prob + english_prob
                logger.info(f"Hinglish detected (Hindi: {hindi_prob:.2f}, English: {english_prob:.2f})")
            elif english_prob > 0.7:
                # Strong English signal
                detected_language = 'en'
                confidence = english_prob
                logger.info(f"Pure English detected (confidence: {confidence:.2f})")
            elif hindi_prob > 0.5:
                # Strong Hindi signal
                detected_language = 'hi'
                confidence = hindi_prob
                logger.info(f"Hindi/Hinglish detected (confidence: {confidence:.2f})")
            else:
                # Unclear - default to Hindi for better Hinglish handling
                detected_language = 'hi'
                confidence = max(probs.values())
                logger.info(f"Language unclear, defaulting to Hindi (confidence: {confidence:.2f})")
            
            return {
                'language': detected_language,
                'confidence': confidence,
                'hindi_prob': hindi_prob,
                'english_prob': english_prob
            }
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return {
                'language': 'hi',  # Default to Hindi for Hinglish compatibility
                'confidence': 0.0,
                'hindi_prob': 0.0,
                'english_prob': 0.0
            }
    
    def split_long_audio(self, audio_path, max_duration=None):
        """
        Split long audio files into chunks (adapted from your WhisperX logic)
        """
        if max_duration is None:
            max_duration = self.config["max_audio_duration"]
            
        logger.info(f"Checking if audio needs splitting: {audio_path}")
        
        try:
            # Get duration using librosa
            duration = librosa.get_duration(path=audio_path)
            
            if duration <= max_duration:
                logger.info(f"Audio duration {duration:.1f}s is within limit. No splitting needed.")
                return [audio_path]
            
            logger.info(f"Audio duration {duration:.1f}s exceeds limit. Splitting into chunks...")
            
            # Load and split audio
            audio_data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            
            samples_per_chunk = int(max_duration * sample_rate)
            total_samples = len(audio_data)
            num_chunks = int(np.ceil(total_samples / samples_per_chunk))
            
            chunk_paths = []
            base_path = os.path.splitext(audio_path)[0]
            file_ext = os.path.splitext(audio_path)[1] or '.wav'
            
            for i in range(num_chunks):
                start_sample = i * samples_per_chunk
                end_sample = min((i + 1) * samples_per_chunk, total_samples)
                
                chunk_data = audio_data[start_sample:end_sample]
                chunk_path = f"{base_path}_chunk_{i+1}{file_ext}"
                
                # Save chunk
                sf.write(chunk_path, chunk_data, sample_rate)
                chunk_paths.append(chunk_path)
                logger.info(f"Created chunk {i+1}/{num_chunks}: {chunk_path}")
            
            return chunk_paths
            
        except Exception as e:
            logger.error(f"Error splitting audio: {e}")
            return [audio_path]  # Return original if splitting fails
    
    def transcribe_for_rag(self, audio_path):
        """
        Main transcription function for RAG - uses your WhisperX intelligence with Whisper
        """
        logger.info(f"Starting intelligent Whisper transcription: {audio_path}")
        
        try:
            # Step 1: Detect language (your WhisperX approach)
            language_result = self.detect_language_from_sample(audio_path)
            detected_language = language_result['language']
            confidence = language_result['confidence']
            
            logger.info(f"Using language: {detected_language} (confidence: {confidence:.2f})")
            
            # Step 2: Check if audio needs splitting
            audio_chunks = self.split_long_audio(audio_path)
            
            all_transcripts = []
            
            # Step 3: Process each chunk with consistent language
            for i, chunk_path in enumerate(audio_chunks):
                logger.info(f"Processing chunk {i+1}/{len(audio_chunks)}: {chunk_path}")
                
                chunk_transcript = self._transcribe_single_chunk(
                    chunk_path, 
                    detected_language
                )
                
                if chunk_transcript:
                    all_transcripts.append(chunk_transcript)
                
                # Clean up chunk files (except original)
                if chunk_path != audio_path and os.path.exists(chunk_path):
                    try:
                        os.remove(chunk_path)
                    except:
                        pass
            
            # Step 4: Combine results
            combined_transcript = " ".join(all_transcripts)
            
            # Step 5: Validate quality
            if not self.validate_transcription_quality(combined_transcript):
                logger.warning("Primary transcription failed quality check, trying fallback")
                return self._fallback_transcription(audio_path, detected_language)
            
            logger.info(f"Transcription successful: {len(combined_transcript)} characters")
            return combined_transcript
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
    
    def _transcribe_single_chunk(self, audio_path, language):
        """Transcribe single chunk with optimal Whisper settings"""
        try:
            # Optimal settings based on detected language
            if language == 'hi':
                # Hinglish-optimized settings
                result = self.whisper_model.transcribe(
                    audio_path,
                    language='hi',
                    word_timestamps=True,
                    verbose=False,
                    temperature=0.2,  # Higher for mixed languages
                    best_of=5,
                    beam_size=5,
                    initial_prompt="यह हिंदी और अंग्रेजी का मिश्रण हो सकता है।"
                )
            else:
                # English-optimized settings
                result = self.whisper_model.transcribe(
                    audio_path,
                    language='en',
                    word_timestamps=True,
                    verbose=False,
                    temperature=0.0,  # Lower for pure language
                    best_of=5,
                    beam_size=5
                )
            
            if not result or not result.get("segments"):
                logger.warning("No segments found in transcription")
                return ""
            
            # Extract clean text (no timestamps for RAG)
            transcript_parts = []
            for segment in result["segments"]:
                text = segment.get("text", "").strip()
                if text and len(text) > 2:
                    transcript_parts.append(text)
            
            return " ".join(transcript_parts)
            
        except Exception as e:
            logger.error(f"Single chunk transcription failed: {e}")
            return ""
    
    def _fallback_transcription(self, audio_path, primary_language):
        """Fallback transcription with opposite language"""
        logger.info("Attempting fallback transcription")
        
        try:
            # Try opposite language
            fallback_language = 'en' if primary_language == 'hi' else 'hi'
            
            result = self.whisper_model.transcribe(
                audio_path,
                language=fallback_language,
                word_timestamps=True,
                temperature=0.3
            )
            
            if result and result.get("segments"):
                transcript_parts = []
                for segment in result["segments"]:
                    text = segment.get("text", "").strip()
                    if text:
                        transcript_parts.append(text)
                
                fallback_transcript = " ".join(transcript_parts)
                
                if self.validate_transcription_quality(fallback_transcript):
                    logger.info(f"Fallback transcription successful with {fallback_language}")
                    return fallback_transcript
            
        except Exception as e:
            logger.warning(f"Fallback transcription failed: {e}")
        
        return ""
    
    def validate_transcription_quality(self, text):
        """Quality validation (your logic adapted)"""
        if not text or len(text.strip()) < self.config["min_transcript_length"]:
            return False
        
        words = text.split()
        if len(words) < 3:
            return False
        
        # Check for repetitive garbage like "aam aam aam"
        word_counts = Counter(words)
        most_common_word, max_count = word_counts.most_common(1)[0]
        
        repetition_ratio = max_count / len(words)
        if repetition_ratio > self.config["quality_threshold"]:
            logger.warning(f"Quality check failed: '{most_common_word}' repeated {repetition_ratio:.1%}")
            return False
        
        # Check for music-only content
        music_indicators = ["♪", "music plays", "intro music"]
        if any(indicator in text.lower() for indicator in music_indicators):
            non_music_words = [w for w in words if not any(ind in w.lower() for ind in music_indicators)]
            if len(non_music_words) < 5:
                logger.warning("Quality check failed: mostly music content")
                return False
        
        logger.info(f"Quality check passed: {len(words)} words, {repetition_ratio:.1%} repetition")
        return True

# Global processor instance
smart_processor = None

def get_smart_processor(whisper_model):
    """Get or create smart processor instance"""
    global smart_processor
    if smart_processor is None:
        smart_processor = SmartWhisperProcessor(whisper_model)
    return smart_processor
import ollama
import os
import uuid
import json
import shutil
import mimetypes
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import faiss
import whisper
import fitz  # PyMuPDF
import sqlite3
from werkzeug.utils import secure_filename
import hashlib
import re
from rank_bm25 import BM25Okapi
import logging
import threading
import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Any, Union
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import sys
import traceback
from datetime import datetime, timezone, timedelta
import tempfile
import platform
from audio_processor import get_smart_processor




# Enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_temp_dir():
    """Get cross-platform temporary directory"""
    if platform.system() == "Windows":
        return os.path.join(os.getcwd(), 'temp')
    else:
        return tempfile.gettempdir()

TEMP_DIR = get_temp_dir()
os.makedirs(TEMP_DIR, exist_ok=True)


# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))

def ist_now():
    return datetime.now(IST).replace(tzinfo=None)
# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring disabled")

# Initialize components with error handling
try:
    whisper_model = whisper.load_model("base")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    whisper_model = None

# Enhanced Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTIONS_DIR = os.path.join(BASE_DIR, 'collections')
os.makedirs(COLLECTIONS_DIR, exist_ok=True)
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)

# Global search database for cross-collection search
GLOBAL_SEARCH_DB = os.path.join(BASE_DIR, 'global_search.db')

# Enhanced embedding dimension
EMBEDDING_DIMENSION = 768

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 100

# Search configuration
DEFAULT_SEARCH_LIMIT = 50
MAX_SEARCH_LIMIT = 1000

MAX_FILE_SIZE_MB = 500
MAX_MEMORY_USAGE_MB = 2048
BATCH_SIZE = 50

# File size limits by type
FILE_SIZE_LIMITS = {
    'pdf': 200 * 1024 * 1024,      # 200MB for PDFs
    'audio': 500 * 1024 * 1024,    # 500MB for audio
    'text': 100 * 1024 * 1024,     # 100MB for text
    'document': 50 * 1024 * 1024   # 50MB for docs
}

# Enhanced file type support
ALLOWED_EXTENSIONS = {
    'audio': {'wav', 'mp3', 'ogg', 'm4a', 'flac', 'aac'},
    'pdf': {'pdf'},
    'text': {'txt', 'md', 'csv', 'json', 'rtf', 'log'},
    'document': {'doc', 'docx', 'odt'},
    'image': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
}

MIME_TYPE_MAP = {
    'audio/mpeg': 'audio',
    'audio/mp3': 'audio',
    'audio/ogg': 'audio',
    'audio/wav': 'audio',
    'audio/x-wav': 'audio',
    'audio/m4a': 'audio',
    'audio/mp4': 'audio',
    'audio/flac': 'audio',
    'audio/aac': 'audio',
    'application/pdf': 'pdf',
    'text/plain': 'text',
    'text/markdown': 'text',
    'text/csv': 'text',
    'application/json': 'text',
    'application/rtf': 'text',
    'application/msword': 'document',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'document',
    'image/png': 'image',
    'image/jpeg': 'image',
    'image/gif': 'image'
}

# Enhanced thread pool for concurrent operations
executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="RAG-Worker")

# Enhanced error handling decorators
def handle_errors(default_return=None, log_error=True):
    """Decorator for consistent error handling"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                if isinstance(default_return, tuple) and len(default_return) == 2:
                    return default_return[0], str(e)
                return default_return
        return wrapper
    return decorator

@contextmanager
def atomic_file_operation(temp_path: str, final_path: str):
    """Context manager for atomic file operations"""
    try:
        yield temp_path
        # Only move if we reach this point (no exceptions)
        shutil.move(temp_path, final_path)
    except Exception as e:
        # Clean up temp file on any error
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise e

@contextmanager
def faiss_index_transaction(index_path: str):
    """Context manager for transactional FAISS operations"""
    backup_path = None
    try:
        # Create backup
        backup_path = index_path + f".backup_{int(time.time())}"
        if os.path.exists(index_path):
            shutil.copy2(index_path, backup_path)
        
        # Load index
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
        else:
            # Create new HNSW index
            index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 100
        
        original_count = index.ntotal
        yield index
        
        # Save if successful
        faiss.write_index(index, index_path)
        
        # Remove backup on success
        if backup_path and os.path.exists(backup_path):
            os.remove(backup_path)
            
    except Exception as e:
        # Restore backup on failure
        if backup_path and os.path.exists(backup_path):
            if os.path.exists(index_path):
                os.remove(index_path)
            shutil.move(backup_path, index_path)
        raise e

class EmbeddingGenerator:
    """Thread-safe embedding generator with retry logic"""
    
    def __init__(self, model_name="nomic-embed-text", max_retries=3, retry_delay=1.0):
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.lock = threading.Lock()
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding with retry logic"""
        for attempt in range(self.max_retries):
            try:
                with self.lock:
                    response = ollama.embeddings(model=self.model_name, prompt=text)
                    embedding = np.array(response["embedding"]).astype('float32')
                    return embedding
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Failed to generate embedding after {self.max_retries} attempts: {e}")
        return None
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[Tuple[int, Optional[np.ndarray]]]:
        """Generate embeddings for batch of texts with threading"""
        results = []
        
        def generate_single(idx_text):
            idx, text = idx_text
            embedding = self.generate_embedding(text)
            return (idx, embedding)
        
        # Use ThreadPoolExecutor for concurrent embedding generation
        with ThreadPoolExecutor(max_workers=3) as pool:
            future_to_idx = {pool.submit(generate_single, (i, text)): i for i, text in enumerate(texts)}
            
            for future in as_completed(future_to_idx):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx = future_to_idx[future]
                    logger.error(f"Error generating embedding for text {idx}: {e}")
                    results.append((idx, None))
        
        # Sort by original index
        results.sort(key=lambda x: x[0])
        return results

# Global embedding generator
embedding_generator = EmbeddingGenerator()

# Global search index manager with enhanced thread safety
class GlobalSearchManager:
    """Enhanced global search manager with thread safety"""
    
    def __init__(self):
        self.db_path = GLOBAL_SEARCH_DB
        self.bm25_cache = {}
        self.tfidf_cache = {}
        self.lock = threading.RLock()  # Reentrant lock
        self._init_global_db()
    
    def _init_global_db(self):
        """Initialize global search database with enhanced schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS global_documents (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    collection_id TEXT,
                    memory_id TEXT,
                    title TEXT,
                    content TEXT,
                    content_hash TEXT,
                    memory_type TEXT,
                    metadata TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            # Enhanced indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON global_documents(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_collection_id ON global_documents(collection_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON global_documents(content_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON global_documents(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON global_documents(status)")
            
            # Full-text search support with enhanced configuration
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    id, title, content, 
                    content='global_documents',
                    content_rowid='rowid',
                    tokenize='porter'
                )
            """)
    
    @handle_errors(default_return=False)
    def index_document(self, user_id, collection_id, memory_id, title, content, memory_type, metadata=None):
        """Add or update document in global search index with validation"""
        if not content or len(content.strip()) < 10:
            logger.warning(f"Skipping document {memory_id} - insufficient content")
            return False
            
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                # Check if document already exists with same content
                cursor = conn.execute(
                    "SELECT content_hash FROM global_documents WHERE id = ?", 
                    (memory_id,)
                )
                existing = cursor.fetchone()
                
                if existing and existing[0] == content_hash:
                    logger.debug(f"Document {memory_id} unchanged, skipping index update")
                    return True
                
                # Insert or update main table
                conn.execute("""
                    INSERT OR REPLACE INTO global_documents 
                    (id, user_id, collection_id, memory_id, title, content, content_hash, 
                     memory_type, metadata, created_at, updated_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory_id, user_id, collection_id, memory_id, title, content, 
                    content_hash, memory_type, json.dumps(metadata or {}), 
                    datetime.now().isoformat(), datetime.now().isoformat(), 'active'
                ))
                
                # Update FTS index
                conn.execute("""
                    INSERT OR REPLACE INTO documents_fts (id, title, content)
                    VALUES (?, ?, ?)
                """, (memory_id, title, content))
        
        # Clear caches
        self.bm25_cache.clear()
        self.tfidf_cache.clear()
        return True
    
    @handle_errors(default_return=False)
    def remove_document(self, memory_id):
        """Remove document from global search index"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                # Soft delete - mark as inactive
                conn.execute(
                    "UPDATE global_documents SET status = 'deleted', updated_at = ? WHERE id = ?", 
                    (datetime.now().isoformat(), memory_id)
                )
                conn.execute("DELETE FROM documents_fts WHERE id = ?", (memory_id,))
        
        self.bm25_cache.clear()
        self.tfidf_cache.clear()
        return True

# Global instance
global_search_manager = GlobalSearchManager()

# Enhanced utility functions
@handle_errors(default_return=None)
def detect_file_type(file):
    """Enhanced file type detection with better accuracy"""
    if not hasattr(file, 'filename') or not file.filename:
        return None
        
    filename = file.filename.lower()
    ext = filename.rsplit('.', 1)[1] if '.' in filename else ''
    
    # Check extension first
    for type_key, extensions in ALLOWED_EXTENSIONS.items():
        if ext in extensions:
            return type_key
    
    # Check MIME type
    mime_type = mimetypes.guess_type(filename)[0]
    if mime_type in MIME_TYPE_MAP:
        return MIME_TYPE_MAP[mime_type]
    
    return None

def allowed_file(filename, file_type):
    """Enhanced file validation with path traversal protection"""
    if not filename or '.' not in filename:
        return False
    
    # Security: Check for path traversal attempts
    if '..' in filename or '/' in filename or '\\' in filename:
        logger.warning(f"Potential path traversal attempt in filename: {filename}")
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    allowed_exts = set()
    
    if file_type:
        allowed_exts = ALLOWED_EXTENSIONS.get(file_type, set())
    else:
        # Allow any supported extension if type not specified
        for exts in ALLOWED_EXTENSIONS.values():
            allowed_exts.update(exts)
    
    return ext in allowed_exts

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Enhanced text chunking with smart boundaries and validation"""
    if not text or not isinstance(text, str):
        return []
    
    text = text.strip()
    text_length = len(text)
    
    if text_length <= chunk_size:
        return [text] if text_length >= MIN_CHUNK_SIZE else []
    
    chunks = []
    start = 0
    
    while start < text_length:
        end = start + chunk_size
        
        if end < text_length:
            # Enhanced boundary detection with multiple fallbacks
            boundaries = [
                ('\n\n', 300),  # Paragraph boundaries
                ('.', 200),     # Sentence boundaries  
                ('!', 200),     # Exclamation boundaries
                ('?', 200),     # Question boundaries
                (';', 150),     # Semicolon boundaries
                ('\n', 100),    # Line boundaries
                (' ', 50)       # Word boundaries
            ]
            
            best_boundary = end
            for boundary_char, search_range in boundaries:
                search_start = max(start, end - search_range)
                boundary_pos = text.rfind(boundary_char, search_start, end)
                if boundary_pos > start:
                    best_boundary = boundary_pos + len(boundary_char)
                    break
            
            end = best_boundary
        
        chunk = text[start:end].strip()
        
        # Enhanced chunk validation
        if len(chunk) >= MIN_CHUNK_SIZE or start + len(chunk) >= text_length:
            if chunk and len(chunk.split()) >= 3:  # At least 3 words
                chunks.append(chunk)
        
        # Enhanced overlap calculation
        start = max(end - overlap, start + 1)
        if start >= text_length:
            break
    
    return chunks

# Enhanced path management functions with validation
def get_user_collections_dir(user_id):
    """Enhanced user directory management with validation"""
    # Validate user_id to prevent path traversal
    if not isinstance(user_id, (int, str)) or str(user_id).strip() == '':
        raise ValueError("Invalid user_id")
    
    # Sanitize user_id
    user_id_str = str(user_id).replace('..', '').replace('/', '').replace('\\', '')
    
    user_dir = os.path.join(COLLECTIONS_DIR, f'user_{user_id_str}')
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def get_collection_path(user_id, collection_id):
    """Enhanced collection path management with validation"""
    if not collection_id or '..' in str(collection_id):
        raise ValueError("Invalid collection_id")
    
    collection_path = os.path.join(get_user_collections_dir(user_id), str(collection_id))
    os.makedirs(collection_path, exist_ok=True)
    return collection_path

def get_collection_metadata_path(user_id, collection_id):
    """Get metadata file path with validation"""
    return os.path.join(get_collection_path(user_id, collection_id), 'metadata.json')

def get_collection_index_path(user_id, collection_id):
    """Enhanced index path with HNSW support"""
    return os.path.join(get_collection_path(user_id, collection_id), 'hnsw_index.faiss')

def get_collection_documents_path(user_id, collection_id):
    """Enhanced documents directory"""
    docs_path = os.path.join(get_collection_path(user_id, collection_id), 'documents')
    os.makedirs(docs_path, exist_ok=True)
    return docs_path

def get_collection_chunks_path(user_id, collection_id):
    """Get chunks database path for this collection"""
    return os.path.join(get_collection_path(user_id, collection_id), 'chunks.db')

# Enhanced memory management functions
def check_memory_usage():
    """Enhanced memory usage monitoring"""
    if not PSUTIL_AVAILABLE:
        return True
    
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        if memory_mb > MAX_MEMORY_USAGE_MB:
            logger.warning(f"High memory usage: {memory_mb:.1f}MB")
            gc.collect()
            
            # Re-check after garbage collection
            memory_mb_after = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory after GC: {memory_mb_after:.1f}MB")
            
        return memory_mb < MAX_MEMORY_USAGE_MB * 1.2  # 20% buffer
    except Exception as e:
        logger.warning(f"Error checking memory usage: {e}")
        return True

def check_file_size(file_path, file_type):
    """Enhanced file size validation"""
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    file_size = os.path.getsize(file_path)
    limit = FILE_SIZE_LIMITS.get(file_type, MAX_FILE_SIZE_MB * 1024 * 1024)
    
    if file_size > limit:
        size_mb = file_size / (1024 * 1024)
        limit_mb = limit / (1024 * 1024)
        return False, f"File size ({size_mb:.1f}MB) exceeds limit ({limit_mb:.1f}MB) for {file_type} files"
    
    return True, None

# Enhanced collection management (keeping original function names)
@handle_errors(default_return=(None, "Error creating collection"))
def create_collection(user_id, name, description=""):
    """Enhanced collection creation with better validation and indexing"""
    if not name or not name.strip():
        return None, "Collection name is required"
    
    collection_id = str(uuid.uuid4())
    
    try:
        user_collections_dir = get_user_collections_dir(user_id)
        collection_path = get_collection_path(user_id, collection_id)
        
        # Create metadata with enhanced fields
        metadata = {
            "id": collection_id,
            "user_id": user_id,
            "name": name.strip(),
            "description": description.strip(),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "memories": [],
            "total_documents": 0,
            "total_chunks": 0,
            "supported_search": ["semantic", "keyword", "hybrid"],
            "index_version": "2.0",
            "status": "active"
        }
        
        # Save metadata atomically
        metadata_path = get_collection_metadata_path(user_id, collection_id)
        temp_metadata_path = metadata_path + '.tmp'
        
        with open(temp_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        shutil.move(temp_metadata_path, metadata_path)
        
        # Initialize enhanced HNSW index
        index_path = get_collection_index_path(user_id, collection_id)
        index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 100
        faiss.write_index(index, index_path)
        
        # Initialize chunks database
        _init_chunks_database(user_id, collection_id)
        
        logger.info(f"Created collection {collection_id} for user {user_id}")
        return collection_id, metadata
        
    except Exception as e:
        # Clean up on failure
        try:
            collection_path = get_collection_path(user_id, collection_id)
            if os.path.exists(collection_path):
                shutil.rmtree(collection_path)
        except:
            pass
        raise e

def _init_chunks_database(user_id, collection_id):
    """Initialize enhanced chunks database for this collection"""
    chunks_db_path = get_collection_chunks_path(user_id, collection_id)
    
    with sqlite3.connect(chunks_db_path) as conn:
        # Enable WAL mode for better performance
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding_index INTEGER,
                embedding_status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(memory_id, chunk_index)
            )
        """)
        
        # Enhanced indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_id ON chunks(memory_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_index ON chunks(chunk_index)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_embedding_index ON chunks(embedding_index)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_embedding_status ON chunks(embedding_status)")

@handle_errors(default_return=[])
def get_all_collections(user_id):
    """Enhanced collection listing with statistics and error handling"""
    collections = []
    user_collections_dir = get_user_collections_dir(user_id)
    
    if not os.path.exists(user_collections_dir):
        return collections
    
    for collection_id in os.listdir(user_collections_dir):
        collection_dir = os.path.join(user_collections_dir, collection_id)
        if not os.path.isdir(collection_dir):
            continue
            
        metadata_path = get_collection_metadata_path(user_id, collection_id)
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    collection_data = json.load(f)
                
                # Add real-time statistics
                collection_data['stats'] = _get_collection_stats(user_id, collection_id)
                collections.append(collection_data)
                
            except Exception as e:
                logger.error(f"Error loading collection {collection_id}: {e}")
                continue
    
    return sorted(collections, key=lambda x: x.get('updated_at', ''), reverse=True)

def _get_collection_stats(user_id, collection_id):
    """Get real-time collection statistics with error handling"""
    stats = {
        "total_memories": 0,
        "total_chunks": 0,
        "memory_types": {},
        "total_size_mb": 0,
        "index_size": 0,
        "processing_status": {}
    }
    
    try:
        # Get memory count and types
        collection = get_collection(user_id, collection_id)
        if collection:
            memories = collection.get("memories", [])
            stats["total_memories"] = len(memories)
            
            for memory in memories:
                mem_type = memory.get("type", "unknown")
                stats["memory_types"][mem_type] = stats["memory_types"].get(mem_type, 0) + 1
        
        # Get chunk count and processing status
        chunks_db_path = get_collection_chunks_path(user_id, collection_id)
        if os.path.exists(chunks_db_path):
            with sqlite3.connect(chunks_db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM chunks")
                stats["total_chunks"] = cursor.fetchone()[0]
                
                # Get processing status counts
                cursor = conn.execute("""
                    SELECT embedding_status, COUNT(*) 
                    FROM chunks 
                    GROUP BY embedding_status
                """)
                for status, count in cursor.fetchall():
                    stats["processing_status"][status] = count
        
        # Get index size
        index_path = get_collection_index_path(user_id, collection_id)
        if os.path.exists(index_path):
            try:
                index = faiss.read_index(index_path)
                stats["index_size"] = index.ntotal
            except:
                stats["index_size"] = 0
        
        # Get directory size
        collection_path = get_collection_path(user_id, collection_id)
        if os.path.exists(collection_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(collection_path):
                for filename in filenames:
                    try:
                        filepath = os.path.join(dirpath, filename)
                        total_size += os.path.getsize(filepath)
                    except:
                        continue
            stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)
    
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
    
    return stats

@handle_errors(default_return=None)
def get_collection(user_id, collection_id):
    """Enhanced collection retrieval with validation and upgrade support"""
    metadata_path = get_collection_metadata_path(user_id, collection_id)
    
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            collection_data = json.load(f)
        
        # Validate and upgrade old collections
        if not collection_data.get("index_version"):
            collection_data = _upgrade_collection(user_id, collection_id, collection_data)
        
        # Add runtime status
        collection_data["last_accessed"] = datetime.now().isoformat()
        
        return collection_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in collection {collection_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading collection {collection_id}: {e}")
        return None

def _upgrade_collection(user_id, collection_id, collection_data):
    """Enhanced collection upgrade with backup"""
    logger.info(f"Upgrading collection {collection_id} to new format")
    
    # Create backup
    metadata_path = get_collection_metadata_path(user_id, collection_id)
    backup_path = metadata_path + f".backup_{int(time.time())}"
    shutil.copy2(metadata_path, backup_path)
    
    try:
        collection_data.update({
            "updated_at": datetime.now().isoformat(),
            "total_documents": len(collection_data.get("memories", [])),
            "total_chunks": 0,
            "supported_search": ["semantic", "keyword", "hybrid"],
            "index_version": "2.0",
            "status": "active"
        })
        
        # Initialize chunks database if not exists
        _init_chunks_database(user_id, collection_id)
        
        # Upgrade index to HNSW if needed
        old_index_path = os.path.join(get_collection_path(user_id, collection_id), 'index')
        new_index_path = get_collection_index_path(user_id, collection_id)
        
        if os.path.exists(old_index_path) and not os.path.exists(new_index_path):
            try:
                old_index = faiss.read_index(old_index_path)
                
                # Create new HNSW index
                new_index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, 32)
                new_index.hnsw.efConstruction = 200
                new_index.hnsw.efSearch = 100
                
                # Transfer vectors if any exist
                if old_index.ntotal > 0:
                    vectors = old_index.reconstruct_n(0, old_index.ntotal)
                    new_index.add(vectors)
                
                faiss.write_index(new_index, new_index_path)
                logger.info(f"Upgraded index for collection {collection_id}")
                
            except Exception as e:
                logger.error(f"Error upgrading index: {e}")
                # Create new empty index as fallback
                index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, 32)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 100
                faiss.write_index(index, new_index_path)
        
        # Save upgraded metadata
        with open(metadata_path, 'w') as f:
            json.dump(collection_data, f, indent=2)
        
        # Remove backup on success
        os.remove(backup_path)
        
        return collection_data
        
    except Exception as e:
        logger.error(f"Error upgrading collection: {e}")
        # Restore backup
        if os.path.exists(backup_path):
            shutil.move(backup_path, metadata_path)
        raise e

@handle_errors(default_return=False)
def delete_collection(user_id, collection_id):
    """Enhanced collection deletion with proper cleanup"""
    try:
        # Remove from global search index first
        collection = get_collection(user_id, collection_id)
        if collection:
            for memory in collection.get("memories", []):
                global_search_manager.remove_document(memory["id"])
        
        # Remove collection directory
        collection_path = get_collection_path(user_id, collection_id)
        if os.path.exists(collection_path):
            shutil.rmtree(collection_path)
        
        logger.info(f"Deleted collection {collection_id} for user {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting collection {collection_id}: {e}")
        return False

# Enhanced text extraction functions with better error handling
@handle_errors(default_return="")
def extract_text_from_pdf(pdf_path):
    """Enhanced PDF text extraction with memory management and validation"""
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return ""
    
    try:
        file_size = os.path.getsize(pdf_path)
        logger.info(f"Processing PDF: {file_size / (1024*1024):.1f}MB")
        
        # Validate file size
        size_ok, error = check_file_size(pdf_path, 'pdf')
        if not size_ok:
            logger.error(f"PDF size validation failed: {error}")
            return ""
        
        doc = fitz.open(pdf_path)
        text_parts = []
        total_pages = len(doc)
        
        if total_pages == 0:
            logger.warning("PDF has no pages")
            doc.close()
            return ""
        
        # Process pages in batches for memory efficiency
        batch_size = 5 if file_size > 50*1024*1024 else 20
        successful_pages = 0
        
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            
            for page_num in range(batch_start, batch_end):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    
                    if page_text and page_text.strip():
                        # Clean and validate page text
                        cleaned_text = page_text.strip()
                        if len(cleaned_text) > 10:  # Minimum content threshold
                            text_parts.append(f"[Page {page_num + 1}]\n{cleaned_text}")
                            successful_pages += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {e}")
                    continue
            
            # Memory management check
            if not check_memory_usage():
                logger.warning("High memory usage during PDF processing")
                gc.collect()
        
        doc.close()
        
        if successful_pages == 0:
            logger.warning(f"No readable content found in PDF: {pdf_path}")
            return ""
        
        extracted_text = "\n\n".join(text_parts)
        logger.info(f"PDF extraction complete: {successful_pages}/{total_pages} pages processed, {len(extracted_text)} characters")
        
        return extracted_text
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

@handle_errors(default_return="")
def extract_text_from_audio(audio_path):
    """Enhanced audio extraction using smart Whisper processing"""
    
    if not whisper_model:
        logger.error("Whisper model not available")
        return ""
        
    audio_path = os.path.normpath(audio_path)
    
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return ""
    
    try:
        file_size = os.path.getsize(audio_path)
        logger.info(f"Processing audio with smart Whisper: {file_size / (1024*1024):.1f}MB")
        
        # Validate file size
        size_ok, error = check_file_size(audio_path, 'audio')
        if not size_ok:
            logger.error(f"Audio size validation failed: {error}")
            return ""
        
        # Use smart processor with your existing Whisper model
        processor = get_smart_processor(whisper_model)
        transcript = processor.transcribe_for_rag(audio_path)
        
        if not transcript:
            logger.error("Smart transcription failed")
            return ""
        
        logger.info("Smart Whisper transcription completed successfully")
        return transcript
        
    except Exception as e:
        logger.error(f"Error in smart audio processing {audio_path}: {e}")
        return ""

@handle_errors(default_return="")
def extract_text_from_text_file(file_path):
    """Enhanced text file reading with encoding detection and validation"""
    if not os.path.exists(file_path):
        logger.error(f"Text file not found: {file_path}")
        return ""
    
    try:
        file_size = os.path.getsize(file_path)
        logger.info(f"Processing text file: {file_size / (1024*1024):.1f}MB")
        
        # Validate file size
        size_ok, error = check_file_size(file_path, 'text')
        if not size_ok:
            logger.error(f"Text file size validation failed: {error}")
            return ""
        
        # Enhanced encoding detection
        encodings = ['utf-8', 'utf-16', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                # For large files, read in chunks to manage memory
                if file_size > 10 * 1024 * 1024:  # 10MB+
                    content_parts = []
                    chunk_size = 1024 * 1024  # 1MB chunks
                    
                    with open(file_path, 'r', encoding=encoding) as f:
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            content_parts.append(chunk)
                            
                            # Memory management
                            if not check_memory_usage():
                                logger.warning("High memory usage during text file processing")
                                gc.collect()
                    
                    content = ''.join(content_parts)
                else:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                
                # Validate content
                if content and content.strip():
                    logger.info(f"Text file read successfully with {encoding} encoding: {len(content)} characters")
                    return content.strip()
                else:
                    logger.warning(f"Text file appears to be empty: {file_path}")
                    return ""
                        
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.error(f"Error reading text file with {encoding}: {e}")
                break
        
        logger.error(f"Could not read text file with any supported encoding: {file_path}")
        return ""
        
    except Exception as e:
        logger.error(f"Error processing text file {file_path}: {e}")
        return ""

# CRITICAL FIX: Enhanced memory processing with atomic operations and proper error handling
@handle_errors(default_return=(None, "Error processing memory"))
def process_memory(user_id, collection_id, file, memory_type, title, description=""):
    """Enhanced memory processing with atomic operations and comprehensive error handling"""
    collection = get_collection(user_id, collection_id)
    if not collection:
        return None, "Collection not found"

    if not file or not hasattr(file, 'filename'):
        return None, "Invalid file provided"

    # Validate inputs
    if not title or not title.strip():
        return None, "Title is required"
    
    if not allowed_file(file.filename, memory_type):
        return None, f"File type not supported for {memory_type}"

    temp_file_path = None
    final_file_path = None
    memory_id = str(uuid.uuid4())
    
    try:
        memory_text = ""

        # Step 1: Secure file handling with atomic operations
        filename = secure_filename(file.filename)
        if not filename:
            return None, "Invalid filename"
        
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        # Create temporary file with unique name
        temp_file_path = os.path.join(TEMP_DIR, f"temp_{memory_id}_{int(time.time())}.{file_ext}")
        
        # Save file to temporary location first
        file.save(temp_file_path)
        
        # Validate file size from saved file
        file_size = os.path.getsize(temp_file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            return None, f"File too large ({file_size_mb:.1f}MB). Maximum size is {MAX_FILE_SIZE_MB}MB"
        
        # Step 2: Extract text with comprehensive error handling
        start_time = time.time()
        processing_metadata = {"extraction_method": memory_type}
        
        try:
            if memory_type == 'audio':
                memory_text = extract_text_from_audio(temp_file_path)
                processing_metadata["transcription"] = True
            elif memory_type == 'pdf':
                memory_text = extract_text_from_pdf(temp_file_path)
                processing_metadata["pdf_extraction"] = True
            elif memory_type == 'text':
                memory_text = extract_text_from_text_file(temp_file_path)
            else:
                # Auto-detect and process
                detected_type = detect_file_type(file)
                if detected_type == 'audio':
                    memory_text = extract_text_from_audio(temp_file_path)
                    processing_metadata["auto_detected"] = "audio"
                elif detected_type == 'pdf':
                    memory_text = extract_text_from_pdf(temp_file_path)
                    processing_metadata["auto_detected"] = "pdf"
                else:
                    memory_text = extract_text_from_text_file(temp_file_path)
                    processing_metadata["auto_detected"] = "text"
        
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return None, f"Failed to extract text from file: {str(e)}"
        
        processing_time = time.time() - start_time
        
        # Validate extracted content
        if not memory_text or len(memory_text.strip()) < 10:
            return None, "Could not extract meaningful text from file"
        
        # Step 3: Enhanced chunking with validation
        text_chunks = chunk_text(memory_text)
        if not text_chunks:
            return None, "Could not create valid chunks from extracted text"
        
        if len(text_chunks) > 1000:
            return None, f"Document too complex: {len(text_chunks)} chunks (max 1000)"
        
        # Step 4: Move file to final location using atomic operation
        memory_dir = get_collection_documents_path(user_id, collection_id)
        saved_filename = f"{memory_id}.{file_ext}"
        final_file_path = os.path.join(memory_dir, saved_filename)
        
        # Use atomic file operation
        with atomic_file_operation(temp_file_path, final_file_path):
            temp_file_path = None  # Prevent cleanup of moved file
        
        # Step 5: CRITICAL FIX - Generate embeddings with proper index tracking
        successful_embeddings = []
        successful_chunks_data = []
        failed_chunks = []
        
        # Generate embeddings in batches
        for batch_start in range(0, len(text_chunks), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(text_chunks))
            batch_chunks = text_chunks[batch_start:batch_end]
            
            # Generate embeddings for this batch
            embedding_results = embedding_generator.generate_batch_embeddings(batch_chunks)
            
            for relative_idx, embedding in embedding_results:
                absolute_idx = batch_start + relative_idx
                current_chunk_text = batch_chunks[relative_idx]
                
                if embedding is not None:
                    successful_embeddings.append(embedding)
                    successful_chunks_data.append({
                        "chunk_id": f"{memory_id}_chunk_{absolute_idx}",
                        "memory_id": memory_id,
                        "chunk_index": absolute_idx,
                        "total_chunks": len(text_chunks),
                        "content": current_chunk_text,
                        "content_hash": hashlib.md5(current_chunk_text.encode()).hexdigest(),
                        "faiss_position": len(successful_embeddings) - 1  # Track actual FAISS position
                    })
                else:
                    failed_chunks.append(absolute_idx)
                    logger.warning(f"Failed to generate embedding for chunk {absolute_idx}")
            
            # Memory management
            if not check_memory_usage():
                gc.collect()
        
        if not successful_embeddings:
            return None, "Failed to generate any embeddings for the document"
        
        # Log embedding success rate
        success_rate = len(successful_embeddings) / len(text_chunks) * 100
        logger.info(f"Embedding generation: {len(successful_embeddings)}/{len(text_chunks)} successful ({success_rate:.1f}%)")
        
        if failed_chunks:
            logger.warning(f"Failed to embed chunks: {failed_chunks}")
        
        # Step 6: CRITICAL FIX - Atomic database and index operations
        chunks_db_path = get_collection_chunks_path(user_id, collection_id)
        index_path = get_collection_index_path(user_id, collection_id)
        
        # Use transactional operations
        with faiss_index_transaction(index_path) as index:
            start_faiss_index = index.ntotal
            
            # Add embeddings to FAISS index
            embeddings_array = np.array(successful_embeddings)
            index.add(embeddings_array)
            
            # Update chunk data with correct FAISS indices
            for i, chunk_data in enumerate(successful_chunks_data):
                chunk_data["embedding_index"] = start_faiss_index + i
                chunk_data["embedding_status"] = "completed"
                chunk_data["created_at"] = datetime.now().isoformat()
                chunk_data["updated_at"] = datetime.now().isoformat()
            
            # Insert chunks into database within the same transaction context
            with sqlite3.connect(chunks_db_path) as conn:
                # Use transaction for database operations
                conn.execute("BEGIN TRANSACTION")
                
                try:
                    for chunk_data in successful_chunks_data:
                        conn.execute("""
                            INSERT INTO chunks 
                            (chunk_id, memory_id, chunk_index, total_chunks, content, content_hash, 
                             embedding_index, embedding_status, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            chunk_data["chunk_id"],
                            chunk_data["memory_id"],
                            chunk_data["chunk_index"],
                            chunk_data["total_chunks"],
                            chunk_data["content"],
                            chunk_data["content_hash"],
                            chunk_data["embedding_index"],
                            chunk_data["embedding_status"],
                            chunk_data["created_at"],
                            chunk_data["updated_at"]
                        ))
                    
                    conn.execute("COMMIT")
                    logger.info(f"Successfully inserted {len(successful_chunks_data)} chunks into database")
                    
                except Exception as e:
                    conn.execute("ROLLBACK")
                    raise e
        
        # Step 7: Save full text content
        text_path = os.path.join(memory_dir, f"{memory_id}.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(memory_text)
        
        # Step 8: Create enhanced memory metadata
        memory_metadata = {
            "id": memory_id,
            "title": title.strip(),
            "description": description.strip(),
            "type": memory_type,
            "filename": saved_filename,
            "original_filename": filename,
            "created_at": datetime.now().isoformat(),
            "processing_metadata": processing_metadata,
            "text_length": len(memory_text),
            "file_size_mb": file_size_mb,
            "processing_time_seconds": processing_time,
            "chunk_count": len(text_chunks),
            "successful_chunks": len(successful_embeddings),
            "failed_chunks": len(failed_chunks),
            "embedding_success_rate": success_rate,
            "status": "completed"
        }
        
        # Step 9: Update collection metadata atomically
        collection["memories"].append(memory_metadata)
        collection["total_documents"] = len(collection["memories"])
        collection["total_chunks"] = collection.get("total_chunks", 0) + len(successful_embeddings)
        collection["updated_at"] = datetime.now().isoformat()
        
        # Save collection metadata atomically
        metadata_path = get_collection_metadata_path(user_id, collection_id)
        temp_metadata_path = metadata_path + '.tmp'
        
        with open(temp_metadata_path, 'w') as f:
            json.dump(collection, f, indent=2)
        shutil.move(temp_metadata_path, metadata_path)
        
        # Step 10: Add to global search index
        global_search_manager.index_document(
            user_id, collection_id, memory_id, title.strip(), memory_text, memory_type, memory_metadata
        )
        
        logger.info(f"Successfully processed memory {memory_id}")
        logger.info(f"- Text length: {len(memory_text):,} characters")
        logger.info(f"- Chunks created: {len(text_chunks)}")
        logger.info(f"- Successful embeddings: {len(successful_embeddings)}")
        logger.info(f"- Processing time: {processing_time:.2f}s")
        
        return memory_metadata, None
    
    except Exception as e:
        logger.error(f"Error processing memory: {e}", exc_info=True)
        
        # Comprehensive cleanup on failure
        cleanup_files = [temp_file_path, final_file_path]
        for file_path in cleanup_files:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up file: {file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Could not clean up file {file_path}: {cleanup_error}")
        
        # Force garbage collection
        gc.collect()
        
        return None, f"Error processing memory: {str(e)}"

# Enhanced query functions with improved search accuracy
@handle_errors(default_return=([], "Error querying collection"))
def query_collection(user_id, collection_id, query_text, top_k=20, search_type="hybrid"):
    """Enhanced collection querying with multiple search strategies and proper error handling"""
    collection = get_collection(user_id, collection_id)
    if not collection or not collection.get("memories"):
        return [], "Collection not found or empty"
    
    if not query_text or not query_text.strip():
        return [], "Query text is required"
    
    try:
        # Semantic search using embeddings
        semantic_results = _semantic_search(user_id, collection_id, query_text.strip(), top_k * 2)
        
        # Keyword search using BM25
        keyword_results = _keyword_search(user_id, collection_id, query_text.strip(), top_k * 2)
        
        # Combine results based on search type
        if search_type == "semantic":
            final_results = semantic_results[:top_k]
        elif search_type == "keyword":
            final_results = keyword_results[:top_k]
        else:  # hybrid
            final_results = _combine_search_results(semantic_results, keyword_results, top_k)
        
        # Get full content for results with enhanced metadata
        memory_dir = get_collection_documents_path(user_id, collection_id)
        relevant_memories = []
        
        for result in final_results:
            chunk_id = result["chunk_id"]
            memory_id = result["memory_id"]
            
            # Get memory metadata
            memory_metadata = None
            for mem in collection["memories"]:
                if mem["id"] == memory_id:
                    memory_metadata = mem
                    break
            
            if memory_metadata:
                # Use chunk content for better relevance
                content = result.get("content", "")
                
                # If chunk content is short, get more context
                if len(content) < 200:
                    try:
                        text_path = os.path.join(memory_dir, f"{memory_id}.txt")
                        if os.path.exists(text_path):
                            with open(text_path, 'r', encoding='utf-8') as f:
                                full_content = f.read()
                                # Use full content if chunk is too short
                                if len(content) < 100:
                                    content = full_content
                    except Exception as e:
                        logger.warning(f"Could not read full content for memory {memory_id}: {e}")
                
                relevant_memories.append({
                    "metadata": memory_metadata,
                    "content": content,
                    "distance": result.get("distance", 0.0),
                    "score": result.get("score", 0.0),
                    "chunk_id": chunk_id,
                    "chunk_index": result.get("chunk_index", 0),
                    "search_type": result.get("search_type", "unknown")
                })
        
        logger.info(f"Query '{query_text}' returned {len(relevant_memories)} results using {search_type} search")
        return relevant_memories, None
    
    except Exception as e:
        logger.error(f"Error querying collection: {e}")
        return [], str(e)

def _semantic_search(user_id, collection_id, query_text, top_k):
    """Enhanced semantic search with proper error handling"""
    try:
        # Generate query embedding with retry logic
        query_embedding = embedding_generator.generate_embedding(query_text)
        if query_embedding is None:
            logger.error("Failed to generate query embedding")
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search FAISS index
        index_path = get_collection_index_path(user_id, collection_id)
        if not os.path.exists(index_path):
            logger.warning(f"Index not found: {index_path}")
            return []
        
        index = faiss.read_index(index_path)
        
        if index.ntotal == 0:
            logger.info("Index is empty")
            return []
        
        k = min(top_k, index.ntotal)
        distances, indices = index.search(query_embedding, k)
        
        # Get chunk information from database
        chunks_db_path = get_collection_chunks_path(user_id, collection_id)
        results = []
        
        if not os.path.exists(chunks_db_path):
            logger.warning(f"Chunks database not found: {chunks_db_path}")
            return []
        
        with sqlite3.connect(chunks_db_path) as conn:
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:
                    continue
                
                # CRITICAL FIX: Use embedding_index to find correct chunk
                cursor = conn.execute("""
                    SELECT chunk_id, memory_id, chunk_index, content, embedding_status 
                    FROM chunks 
                    WHERE embedding_index = ? AND embedding_status = 'completed'
                """, (int(idx),))
                
                row = cursor.fetchone()
                if row:
                    chunk_id, memory_id, chunk_index, content, status = row
                    results.append({
                        "chunk_id": chunk_id,
                        "memory_id": memory_id,
                        "chunk_index": chunk_index,
                        "content": content,
                        "distance": float(distance),
                        "score": 1.0 / (1.0 + float(distance)),  # Convert distance to score
                        "search_type": "semantic"
                    })
        
        logger.debug(f"Semantic search found {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return []

def _keyword_search(user_id, collection_id, query_text, top_k):
    """Enhanced keyword search using BM25 with caching"""
    try:
        chunks_db_path = get_collection_chunks_path(user_id, collection_id)
        
        if not os.path.exists(chunks_db_path):
            return []
        
        with sqlite3.connect(chunks_db_path) as conn:
            cursor = conn.execute("""
                SELECT chunk_id, memory_id, chunk_index, content 
                FROM chunks 
                WHERE embedding_status = 'completed'
                ORDER BY memory_id, chunk_index
            """)
            chunks = cursor.fetchall()
        
        if not chunks:
            return []
        
        # Prepare documents for BM25
        documents = []
        chunk_info = []
        
        for chunk_id, memory_id, chunk_index, content in chunks:
            # Enhanced tokenization with preprocessing
            content_clean = re.sub(r'[^\w\s]', ' ', content.lower())
            tokens = [token for token in content_clean.split() if len(token) > 2]
            documents.append(tokens)
            chunk_info.append({
                "chunk_id": chunk_id,
                "memory_id": memory_id,
                "chunk_index": chunk_index,
                "content": content
            })
        
        # Perform BM25 search
        bm25 = BM25Okapi(documents)
        query_clean = re.sub(r'[^\w\s]', ' ', query_text.lower())
        query_tokens = [token for token in query_clean.split() if len(token) > 2]
        
        if not query_tokens:
            return []
        
        scores = bm25.get_scores(query_tokens)
        
        # Get top results
        scored_results = []
        for i, score in enumerate(scores):
            if score > 0:
                chunk_data = chunk_info[i].copy()
                chunk_data.update({
                    "score": float(score),
                    "distance": 1.0 / (1.0 + float(score)),
                    "search_type": "keyword"
                })
                scored_results.append(chunk_data)
        
        # Sort by score and return top k
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        result = scored_results[:top_k]
        
        logger.debug(f"Keyword search found {len(result)} results")
        return result
        
    except Exception as e:
        logger.error(f"Error in keyword search: {e}")
        return []

def _combine_search_results(semantic_results, keyword_results, top_k):
    """Enhanced hybrid search result combination with normalization"""
    if not semantic_results and not keyword_results:
        return []
    
    if not semantic_results:
        return keyword_results[:top_k]
    
    if not keyword_results:
        return semantic_results[:top_k]
    
    # Normalize scores for fair combination
    def normalize_scores(results, score_key="score"):
        if not results:
            return results
        scores = [r[score_key] for r in results]
        if max(scores) == min(scores):
            return results
        
        min_score, max_score = min(scores), max(scores)
        for result in results:
            result[f"normalized_{score_key}"] = (result[score_key] - min_score) / (max_score - min_score)
        return results
    
    # Normalize both result sets
    semantic_results = normalize_scores(semantic_results)
    keyword_results = normalize_scores(keyword_results)
    
    # Create a mapping of chunk_id to results
    semantic_map = {r["chunk_id"]: r for r in semantic_results}
    keyword_map = {r["chunk_id"]: r for r in keyword_results}
    
    # Get all unique chunk IDs
    all_chunk_ids = set(semantic_map.keys()) | set(keyword_map.keys())
    
    combined_results = []
    
    for chunk_id in all_chunk_ids:
        semantic_result = semantic_map.get(chunk_id)
        keyword_result = keyword_map.get(chunk_id)
        
        # Calculate hybrid score with normalized values
        semantic_score = semantic_result.get("normalized_score", 0) if semantic_result else 0
        keyword_score = keyword_result.get("normalized_score", 0) if keyword_result else 0
        
        # Weighted combination (semantic slightly favored)
        hybrid_score = 0.6 * semantic_score + 0.4 * keyword_score
        
        # Use the result with more information
        base_result = semantic_result or keyword_result
        base_result = base_result.copy()  # Don't modify original
        
        base_result.update({
            "score": hybrid_score,
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "search_type": "hybrid"
        })
        
        combined_results.append(base_result)
    
    # Sort by hybrid score and return top k
    combined_results.sort(key=lambda x: x["score"], reverse=True)
    return combined_results[:top_k]

@handle_errors(default_return=([], "Error in cross-collection search"))
def query_across_collections(user_id, query_text, collection_ids=None, top_k=50, search_type="hybrid"):
    """Enhanced cross-collection search with parallel processing"""
    if not query_text or not query_text.strip():
        return [], "Query text is required"
    
    try:
        if collection_ids is None:
            # Search all user collections
            collections = get_all_collections(user_id)
            collection_ids = [c["id"] for c in collections if c.get("status") == "active"]
        
        if not collection_ids:
            return [], "No collections found"
        
        all_results = []
        
        # Use ThreadPoolExecutor for parallel search across collections
        def search_collection(collection_id):
            try:
                results, error = query_collection(user_id, collection_id, query_text, top_k, search_type)
                if not error and results:
                    # Add collection info to results
                    for result in results:
                        result["source_collection_id"] = collection_id
                    return results
            except Exception as e:
                logger.error(f"Error searching collection {collection_id}: {e}")
            return []
        
        # Execute searches in parallel
        with ThreadPoolExecutor(max_workers=min(len(collection_ids), 5)) as executor:
            future_to_collection = {executor.submit(search_collection, cid): cid for cid in collection_ids}
            
            for future in as_completed(future_to_collection):
                collection_id = future_to_collection[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Error processing results from collection {collection_id}: {e}")
        
        # Sort all results by score and remove duplicates
        seen_chunks = set()
        unique_results = []
        
        for result in sorted(all_results, key=lambda x: x.get("score", 0), reverse=True):
            chunk_key = (result["memory_id"], result.get("chunk_index", 0))
            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                unique_results.append(result)
        
        final_results = unique_results[:top_k]
        logger.info(f"Cross-collection search returned {len(final_results)} unique results from {len(collection_ids)} collections")
        
        return final_results, None
        
    except Exception as e:
        logger.error(f"Error in cross-collection search: {e}")
        return [], str(e)

@handle_errors(default_return=([], "Error in global search"))
def global_search(user_id, query_text, search_type="hybrid", limit=100):
    """Enhanced global search across all user documents with FTS and semantic search"""
    if not query_text or not query_text.strip():
        return [], "Query text is required"
    
    try:
        results = []
        
        # Full-text search using SQLite FTS
        with sqlite3.connect(global_search_manager.db_path) as conn:
            if search_type in ["keyword", "hybrid"]:
                # Enhanced FTS query with ranking
                fts_query = query_text.strip().replace('"', '""')  # Escape quotes
                
                cursor = conn.execute("""
                    SELECT gd.id, gd.collection_id, gd.title, gd.content, gd.memory_type, 
                           gd.metadata, bm25(documents_fts) as rank
                    FROM documents_fts 
                    JOIN global_documents gd ON documents_fts.id = gd.id
                    WHERE documents_fts MATCH ? AND gd.user_id = ? AND gd.status = 'active'
                    ORDER BY rank
                    LIMIT ?
                """, (fts_query, user_id, limit))
                
                keyword_results = cursor.fetchall()
                
                for row in keyword_results:
                    doc_id, coll_id, title, content, mem_type, metadata_str, rank = row
                    
                    # Truncate content for display
                    display_content = content[:500] + "..." if len(content) > 500 else content
                    
                    try:
                        metadata = json.loads(metadata_str) if metadata_str else {}
                    except:
                        metadata = {}
                    
                    results.append({
                        "document_id": doc_id,
                        "collection_id": coll_id,
                        "title": title,
                        "content": display_content,
                        "full_content": content,  # Keep full content for semantic search
                        "memory_type": mem_type,
                        "metadata": metadata,
                        "score": float(rank) if rank else 0.0,
                        "search_type": "keyword_global"
                    })
        
        # Enhanced semantic search for hybrid mode
        if search_type in ["semantic", "hybrid"] and results:
            # Generate query embedding
            query_embedding = embedding_generator.generate_embedding(query_text)
            
            if query_embedding is not None:
                # Score results using semantic similarity
                for result in results[:20]:  # Limit to top keyword results for efficiency
                    try:
                        # Generate embedding for document content
                        doc_embedding = embedding_generator.generate_embedding(result["full_content"][:1000])
                        
                        if doc_embedding is not None:
                            # Calculate cosine similarity
                            similarity = np.dot(query_embedding, doc_embedding) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                            )
                            result["semantic_score"] = float(similarity)
                            
                            if search_type == "hybrid":
                                # Combine scores
                                result["score"] = 0.6 * result["score"] + 0.4 * result["semantic_score"]
                        
                    except Exception as e:
                        logger.warning(f"Error calculating semantic score for document {result['document_id']}: {e}")
                        result["semantic_score"] = 0.0
        
        # Clean up full_content from results
        for result in results:
            if "full_content" in result:
                del result["full_content"]
        
        # Sort by final score
        results.sort(key=lambda x: x["score"], reverse=True)
        final_results = results[:limit]
        
        logger.info(f"Global search returned {len(final_results)} results")
        return final_results, None
        
    except Exception as e:
        logger.error(f"Error in global search: {e}")
        return [], str(e)

@handle_errors(default_return="I don't have any relevant information to answer your question.")
def generate_response(query, relevant_memories, conversation_history=None):
    """Enhanced response generation with better context handling and validation"""
    if not relevant_memories:
        return "I don't have any relevant memories to answer your question. Try rephrasing your query or check if you have uploaded related documents."
    
    try:
        # Prepare enhanced context with validation
        context_parts = []
        source_info = []
        
        for i, memory in enumerate(relevant_memories[:5]):  # Limit to top 5 for context
            if not isinstance(memory, dict) or 'metadata' not in memory or 'content' not in memory:
                logger.warning(f"Invalid memory format at index {i}")
                continue
            
            metadata = memory['metadata']
            content = memory['content']
            
            if not content or not content.strip():
                logger.warning(f"Empty content for memory {metadata.get('id', 'unknown')}")
                continue
            
            # Truncate very long content
            if len(content) > 1000:
                content = content[:1000] + "..."
            
            source_type = metadata.get('type', 'unknown')
            title = metadata.get('title', 'Untitled')
            filename = metadata.get('original_filename', 'Unknown file')
            score = memory.get('score', 0)
            
            context_parts.append(f"""
Source {i+1}: {title}
Type: {source_type} ({filename})
Relevance Score: {score:.3f}
Content: {content}
""")
            
            source_info.append({
                "title": title,
                "type": source_type,
                "filename": filename,
                "score": score
            })
        
        if not context_parts:
            return "I found some relevant documents but couldn't extract readable content from them."
        
        context = "\n".join(context_parts)
        
        # Build enhanced prompt with conversation history
        conversation_section = ""
        if conversation_history and conversation_history.strip():
            conversation_section = f"""
Previous conversation context:
{conversation_history}

"""
        
        prompt = f"""You are an intelligent AI assistant with access to a user's personal knowledge base. Your task is to provide accurate, helpful responses based on the retrieved information while being transparent about your sources.

{conversation_section}Based on the following retrieved information from the user's documents, please answer their question comprehensively:

{context}

User Question: {query}

Instructions:
1. Provide a clear, direct answer to the user's question
2. Reference specific sources when you use information from them
3. If information from multiple sources is relevant, synthesize it coherently
4. If the retrieved information doesn't fully answer the question, be honest about limitations
5. Maintain a helpful, conversational tone
6. If referencing conversation history, acknowledge previous context appropriately
7. Be concise but thorough

Your response:"""

        # Generate response with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                output = ollama.generate(
                    model="llama3",
                    prompt=prompt,
                    options={
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 1000,
                        "stop": ["User Question:", "Instructions:"]
                    }
                )
                
                response = output.get('response', '').strip()
                
                if not response:
                    raise ValueError("Empty response from LLM")
                
                # Add source information if multiple sources were used
                if len(source_info) > 1:
                    sources_text = "\n\nSources referenced:"
                    for i, info in enumerate(source_info):
                        sources_text += f"\n{i+1}. {info['title']} ({info['type']}) - Relevance: {info['score']:.2f}"
                    response += sources_text
                
                logger.info(f"Generated response of {len(response)} characters using {len(relevant_memories)} sources")
                return response
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to generate response: {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)  # Brief delay before retry
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"I encountered an error while processing your question. Please try again. (Error: {str(e)})"

@handle_errors(default_return=(None, "Error querying specific memory"))
def query_specific_memory(user_id, collection_id, memory_id, query_text):
    """Enhanced specific memory querying with chunk-level search and validation"""
    collection = get_collection(user_id, collection_id)
    if not collection:
        return None, "Collection not found"
    
    if not query_text or not query_text.strip():
        return None, "Query text is required"
    
    # Find the memory metadata with validation
    memory_metadata = None
    for mem in collection.get("memories", []):
        if mem["id"] == memory_id:
            memory_metadata = mem
            break
    
    if not memory_metadata:
        return None, "Memory not found"
    
    try:
        # Get all chunks for this memory with semantic scoring
        chunks_db_path = get_collection_chunks_path(user_id, collection_id)
        
        if not os.path.exists(chunks_db_path):
            # Fallback to full text if chunks DB doesn't exist
            return _query_memory_fallback(user_id, collection_id, memory_id, memory_metadata)
        
        with sqlite3.connect(chunks_db_path) as conn:
            cursor = conn.execute("""
                SELECT chunk_id, chunk_index, content 
                FROM chunks 
                WHERE memory_id = ? AND embedding_status = 'completed'
                ORDER BY chunk_index
            """, (memory_id,))
            
            chunks = cursor.fetchall()
        
        if not chunks:
            # Fallback to full text
            return _query_memory_fallback(user_id, collection_id, memory_id, memory_metadata)
        
        # Enhanced chunk scoring with both keyword and position weighting
        scored_chunks = []
        query_tokens = set(re.findall(r'\b\w+\b', query_text.lower()))
        
        for chunk_id, chunk_index, content in chunks:
            # Keyword matching score
            content_tokens = set(re.findall(r'\b\w+\b', content.lower()))
            keyword_overlap = len(query_tokens & content_tokens) / len(query_tokens) if query_tokens else 0
            
            # Position weighting (earlier chunks slightly favored)
            position_weight = 1.0 - (chunk_index * 0.1)  # Gradual decrease
            position_weight = max(position_weight, 0.5)  # Minimum weight
            
            final_score = keyword_overlap * position_weight
            
            scored_chunks.append({
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "content": content,
                "keyword_score": keyword_overlap,
                "position_weight": position_weight,
                "final_score": final_score
            })
        
        # Sort by final score and take top relevant chunks
        scored_chunks.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Select chunks based on score threshold or minimum count
        top_chunks = []
        score_threshold = 0.1
        
        for chunk in scored_chunks:
            if chunk["final_score"] > score_threshold or len(top_chunks) < 2:
                top_chunks.append(chunk)
            if len(top_chunks) >= 5:  # Maximum chunks to include
                break
        
        # If no chunks meet threshold, take first 2 chunks
        if not top_chunks:
            top_chunks = scored_chunks[:2]
        
        # Combine chunk contents with context markers
        combined_content = "\n\n".join([
            f"[Section {c['chunk_index'] + 1}] {c['content']}"
            for c in top_chunks
        ])
        
        result = [{
            "metadata": memory_metadata,
            "content": combined_content,
            "distance": 0.0,
            "score": 1.0,
            "chunks_used": len(top_chunks),
            "chunk_details": [
                {"index": c["chunk_index"], "score": c["final_score"]} 
                for c in top_chunks
            ]
        }]
        
        logger.info(f"Queried memory {memory_id}: used {len(top_chunks)} chunks")
        return result, None
        
    except Exception as e:
        logger.error(f"Error querying specific memory: {e}")
        return None, str(e)

def _query_memory_fallback(user_id, collection_id, memory_id, memory_metadata):
    """Fallback method for querying memory when chunks are not available"""
    try:
        memory_dir = get_collection_documents_path(user_id, collection_id)
        text_path = os.path.join(memory_dir, f"{memory_id}.txt")
        
        if not os.path.exists(text_path):
            return None, "Memory content file not found"
        
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            return None, "Memory content is empty"
        
        return [{
            "metadata": memory_metadata,
            "content": content,
            "distance": 0.0,
            "score": 1.0,
            "fallback_method": True
        }], None
        
    except Exception as e:
        logger.error(f"Error in memory fallback query: {e}")
        return None, str(e)

@handle_errors(default_return=(False, "Error deleting memory"))
def delete_memory(user_id, collection_id, memory_id):
    """Enhanced memory deletion with comprehensive cleanup and validation"""
    try:
        collection = get_collection(user_id, collection_id)
        if not collection:
            return False, "Collection not found"
        
        # Find memory in collection
        memory_index = None
        memory = None
        for i, mem in enumerate(collection.get("memories", [])):
            if mem["id"] == memory_id:
                memory_index = i
                memory = mem
                break
        
        if memory_index is None:
            return False, "Memory not found"
        
        logger.info(f"Deleting memory {memory_id} from collection {collection_id}")
        
        # Step 1: Remove from global search index
        global_search_manager.remove_document(memory_id)
        
        # Step 2: Get embedding indices before removing chunks
        chunks_db_path = get_collection_chunks_path(user_id, collection_id)
        embedding_indices = []
        
        if os.path.exists(chunks_db_path):
            with sqlite3.connect(chunks_db_path) as conn:
                cursor = conn.execute(
                    "SELECT embedding_index FROM chunks WHERE memory_id = ? AND embedding_index IS NOT NULL", 
                    (memory_id,)
                )
                embedding_indices = [row[0] for row in cursor.fetchall()]
                
                # Remove chunks from database
                conn.execute("DELETE FROM chunks WHERE memory_id = ?", (memory_id,))
                conn.commit()
        
        # Step 3: Remove files with error handling
        memory_dir = get_collection_documents_path(user_id, collection_id)
        files_to_remove = [
            os.path.join(memory_dir, memory["filename"]),  # Original file
            os.path.join(memory_dir, f"{memory_id}.txt")   # Text file
        ]
        
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Removed file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not remove file {file_path}: {e}")
        
        # Step 4: Update collection metadata
        collection["memories"].pop(memory_index)
        collection["total_documents"] = len(collection["memories"])
        collection["updated_at"] = datetime.now().isoformat()
        
        # Recalculate total chunks
        if os.path.exists(chunks_db_path):
            with sqlite3.connect(chunks_db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM chunks")
                collection["total_chunks"] = cursor.fetchone()[0]
        
        # Save updated collection metadata atomically
        metadata_path = get_collection_metadata_path(user_id, collection_id)
        temp_metadata_path = metadata_path + '.tmp'
        
        with open(temp_metadata_path, 'w') as f:
            json.dump(collection, f, indent=2)
        shutil.move(temp_metadata_path, metadata_path)
        
        # Step 5: Rebuild FAISS index to remove deleted embeddings
        # Only rebuild if there were embeddings to remove
        if embedding_indices:
            rebuild_success = rebuild_collection_index(user_id, collection_id)
            if not rebuild_success:
                logger.warning(f"Failed to rebuild index after deleting memory {memory_id}")
        
        logger.info(f"Successfully deleted memory {memory_id}")
        return True, None
        
    except Exception as e:
        logger.error(f"Error deleting memory {memory_id}: {e}")
        return False, str(e)

@handle_errors(default_return=False)
def rebuild_collection_index(user_id, collection_id):
    """Enhanced index rebuilding with comprehensive error handling and validation"""
    try:
        collection = get_collection(user_id, collection_id)
        if not collection:
            logger.error(f"Collection {collection_id} not found for rebuild")
            return False
        
        logger.info(f"Rebuilding index for collection {collection_id}")
        
        # Step 1: Create new HNSW index
        index_path = get_collection_index_path(user_id, collection_id)
        
        # Create backup of existing index
        backup_path = None
        if os.path.exists(index_path):
            backup_path = index_path + f".rebuild_backup_{int(time.time())}"
            shutil.copy2(index_path, backup_path)
        
        try:
            # Create new empty index
            new_index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, 32)
            new_index.hnsw.efConstruction = 200
            new_index.hnsw.efSearch = 100
            
            # Step 2: Get all chunks and regenerate embeddings
            chunks_db_path = get_collection_chunks_path(user_id, collection_id)
            
            if not os.path.exists(chunks_db_path):
                logger.warning(f"Chunks database not found: {chunks_db_path}")
                # Save empty index
                faiss.write_index(new_index, index_path)
                return True
            
            with sqlite3.connect(chunks_db_path) as conn:
                cursor = conn.execute("""
                    SELECT chunk_id, content, memory_id, chunk_index
                    FROM chunks 
                    WHERE embedding_status = 'completed'
                    ORDER BY memory_id, chunk_index
                """)
                chunks = cursor.fetchall()
                
                if not chunks:
                    logger.info("No chunks found, saving empty index")
                    faiss.write_index(new_index, index_path)
                    return True
                
                logger.info(f"Rebuilding index with {len(chunks)} chunks")
                
                # Process chunks in batches
                successful_embeddings = []
                chunk_updates = []
                
                for batch_start in range(0, len(chunks), BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, len(chunks))
                    batch_chunks = chunks[batch_start:batch_end]
                    
                    batch_texts = [chunk[1] for chunk in batch_chunks]  # content
                    embedding_results = embedding_generator.generate_batch_embeddings(batch_texts)
                    
                    for relative_idx, embedding in embedding_results:
                        if embedding is not None:
                            chunk_id, content, memory_id, chunk_index = batch_chunks[relative_idx]
                            
                            successful_embeddings.append(embedding)
                            new_embedding_index = len(successful_embeddings) - 1
                            
                            chunk_updates.append((
                                new_embedding_index,
                                datetime.now().isoformat(),
                                chunk_id
                            ))
                        else:
                            chunk_id = batch_chunks[relative_idx][0]
                            logger.warning(f"Failed to regenerate embedding for chunk {chunk_id}")
                    
                    # Memory management
                    if not check_memory_usage():
                        gc.collect()
                
                # Step 3: Add embeddings to new index
                if successful_embeddings:
                    embeddings_array = np.array(successful_embeddings)
                    new_index.add(embeddings_array)
                    logger.info(f"Added {len(successful_embeddings)} embeddings to new index")
                
                # Step 4: Update database with new embedding indices
                if chunk_updates:
                    conn.execute("BEGIN TRANSACTION")
                    try:
                        conn.executemany("""
                            UPDATE chunks 
                            SET embedding_index = ?, updated_at = ?
                            WHERE chunk_id = ?
                        """, chunk_updates)
                        conn.execute("COMMIT")
                        logger.info(f"Updated {len(chunk_updates)} chunk embedding indices")
                    except Exception as e:
                        conn.execute("ROLLBACK")
                        raise e
            
            # Step 5: Save the new index
            faiss.write_index(new_index, index_path)
            
            # Remove backup on success
            if backup_path and os.path.exists(backup_path):
                os.remove(backup_path)
            
            logger.info(f"Successfully rebuilt index for collection {collection_id} with {new_index.ntotal} vectors")
            return True
            
        except Exception as e:
            # Restore backup on failure
            if backup_path and os.path.exists(backup_path):
                if os.path.exists(index_path):
                    os.remove(index_path)
                shutil.move(backup_path, index_path)
                logger.info("Restored index backup after rebuild failure")
            raise e
        
    except Exception as e:
        logger.error(f"Error rebuilding collection index: {e}")
        return False

# Enhanced utility functions
@handle_errors(default_return=None)
def get_collection_statistics(user_id, collection_id):
    """Enhanced collection statistics with comprehensive metrics"""
    try:
        collection = get_collection(user_id, collection_id)
        if not collection:
            return None
        
        stats = {
            "basic_info": {
                "total_memories": len(collection.get("memories", [])),
                "total_chunks": 0,
                "collection_size_mb": 0,
                "created_at": collection.get("created_at"),
                "updated_at": collection.get("updated_at")
            },
            "memory_types": {},
            "processing_status": {},
            "search_index": {
                "index_size": 0,
                "index_file_size_mb": 0
            },
            "health_metrics": {
                "integrity_score": 100,
                "issues": []
            }
        }
        
        # Memory type distribution
        for memory in collection.get("memories", []):
            mem_type = memory.get("type", "unknown")
            stats["memory_types"][mem_type] = stats["memory_types"].get(mem_type, 0) + 1
        
        # Chunk and processing statistics
        chunks_db_path = get_collection_chunks_path(user_id, collection_id)
        if os.path.exists(chunks_db_path):
            with sqlite3.connect(chunks_db_path) as conn:
                # Total chunks
                cursor = conn.execute("SELECT COUNT(*) FROM chunks")
                stats["basic_info"]["total_chunks"] = cursor.fetchone()[0]
                
                # Processing status distribution
                cursor = conn.execute("""
                    SELECT embedding_status, COUNT(*) 
                    FROM chunks 
                    GROUP BY embedding_status
                """)
                for status, count in cursor.fetchall():
                    stats["processing_status"][status] = count
                
                # Health check - orphaned chunks
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM chunks 
                    WHERE embedding_index IS NULL OR embedding_status != 'completed'
                """)
                orphaned_chunks = cursor.fetchone()[0]
                if orphaned_chunks > 0:
                    stats["health_metrics"]["issues"].append(f"{orphaned_chunks} chunks with missing embeddings")
                    stats["health_metrics"]["integrity_score"] -= min(20, orphaned_chunks * 2)
        
        # Index statistics
        index_path = get_collection_index_path(user_id, collection_id)
        if os.path.exists(index_path):
            try:
                index = faiss.read_index(index_path)
                stats["search_index"]["index_size"] = index.ntotal
                stats["search_index"]["index_file_size_mb"] = round(
                    os.path.getsize(index_path) / (1024 * 1024), 2
                )
                
                # Validate index consistency
                expected_embeddings = stats["processing_status"].get("completed", 0)
                if index.ntotal != expected_embeddings:
                    stats["health_metrics"]["issues"].append(
                        f"Index size mismatch: {index.ntotal} vectors vs {expected_embeddings} completed chunks"
                    )
                    stats["health_metrics"]["integrity_score"] -= 15
                    
            except Exception as e:
                stats["health_metrics"]["issues"].append(f"Index read error: {str(e)}")
                stats["health_metrics"]["integrity_score"] -= 25
        
        # Collection size calculation
        collection_path = get_collection_path(user_id, collection_id)
        if os.path.exists(collection_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(collection_path):
                for filename in filenames:
                    try:
                        filepath = os.path.join(dirpath, filename)
                        total_size += os.path.getsize(filepath)
                    except:
                        continue
            stats["basic_info"]["collection_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        # Final health score adjustment
        stats["health_metrics"]["integrity_score"] = max(0, stats["health_metrics"]["integrity_score"])
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting collection statistics: {e}")
        return None

@handle_errors(default_return=[])
def search_suggestions(user_id, partial_query, limit=10):
    """Enhanced search suggestions with improved relevance and caching"""
    if not partial_query or len(partial_query.strip()) < 2:
        return []
    
    try:
        suggestions = []
        partial_query = partial_query.strip().lower()
        
        # Search in global index for matching titles and content
        with sqlite3.connect(global_search_manager.db_path) as conn:
            # Title-based suggestions (higher priority)
            cursor = conn.execute("""
                SELECT DISTINCT title, memory_type, collection_id
                FROM global_documents 
                WHERE user_id = ? AND status = 'active' AND title LIKE ?
                ORDER BY length(title)
                LIMIT ?
            """, (user_id, f"%{partial_query}%", limit // 2))
            
            for title, mem_type, coll_id in cursor.fetchall():
                suggestions.append({
                    "text": title,
                    "type": "document_title",
                    "memory_type": mem_type,
                    "collection_id": coll_id,
                    "priority": 1
                })
            
            # Content-based suggestions if we need more
            remaining_limit = limit - len(suggestions)
            if remaining_limit > 0:
                cursor = conn.execute("""
                    SELECT DISTINCT title, memory_type, collection_id
                    FROM global_documents 
                    WHERE user_id = ? AND status = 'active' AND content LIKE ? AND title NOT LIKE ?
                    ORDER BY length(title)
                    LIMIT ?
                """, (user_id, f"%{partial_query}%", f"%{partial_query}%", remaining_limit))
                
                for title, mem_type, coll_id in cursor.fetchall():
                    suggestions.append({
                        "text": title,
                        "type": "document_content",
                        "memory_type": mem_type,
                        "collection_id": coll_id,
                        "priority": 2
                    })
        
        # Add common search patterns if we still have room
        if len(suggestions) < limit:
            common_patterns = [
                f"What is {partial_query}",
                f"How to {partial_query}",
                f"When did {partial_query}",
                f"Where is {partial_query}",
                f"Why {partial_query}",
                f"Find {partial_query}"
            ]
            
            for pattern in common_patterns:
                if len(suggestions) >= limit:
                    break
                suggestions.append({
                    "text": pattern,
                    "type": "query_pattern",
                    "priority": 3
                })
        
        # Sort by priority and return
        suggestions.sort(key=lambda x: (x.get("priority", 99), len(x["text"])))
        return suggestions[:limit]
        
    except Exception as e:
        logger.error(f"Error getting search suggestions: {e}")
        return []

# Enhanced system health functions
def validate_system_integrity(user_id=None):
    """Comprehensive system integrity validation"""
    issues = []
    
    try:
        # Check global database
        if not os.path.exists(GLOBAL_SEARCH_DB):
            issues.append("Global search database missing")
        else:
            try:
                with sqlite3.connect(GLOBAL_SEARCH_DB) as conn:
                    conn.execute("SELECT COUNT(*) FROM global_documents").fetchone()
            except Exception as e:
                issues.append(f"Global database corrupted: {e}")
        
        # Check specific user if provided
        if user_id:
            collections = get_all_collections(user_id)
            for collection in collections:
                collection_id = collection["id"]
                
                # Check collection files
                index_path = get_collection_index_path(user_id, collection_id)
                chunks_path = get_collection_chunks_path(user_id, collection_id)
                
                if not os.path.exists(index_path):
                    issues.append(f"Missing index for collection {collection_id}")
                
                if not os.path.exists(chunks_path):
                    issues.append(f"Missing chunks database for collection {collection_id}")
                
                # Validate index consistency
                try:
                    index = faiss.read_index(index_path)
                    with sqlite3.connect(chunks_path) as conn:
                        cursor = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding_status = 'completed'")
                        completed_chunks = cursor.fetchone()[0]
                        
                        if index.ntotal != completed_chunks:
                            issues.append(f"Index mismatch in collection {collection_id}: {index.ntotal} vs {completed_chunks}")
                            
                except Exception as e:
                    issues.append(f"Error validating collection {collection_id}: {e}")
        
        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "checked_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error validating system integrity: {e}")
        return {
            "healthy": False,
            "issues": [f"Validation error: {e}"],
            "checked_at": datetime.now().isoformat()
        }

def cleanup_system_resources():
    """Enhanced system cleanup with comprehensive resource management"""
    cleaned_resources = {
        "temp_files": 0,
        "backup_files": 0,
        "orphaned_files": 0,
        "memory_freed_mb": 0
    }
    
    try:
        # Clean up temporary files
        current_time = time.time()
        
        if os.path.exists(TEMP_DIR):
            for filename in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, filename)
                
                try:
                    # Remove files older than 1 hour
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getctime(file_path)
                        if file_age > 3600:  # 1 hour
                            file_size = os.path.getsize(file_path) / (1024 * 1024)
                            os.remove(file_path)
                            cleaned_resources["temp_files"] += 1
                            cleaned_resources["memory_freed_mb"] += file_size
                except Exception as e:
                    logger.warning(f"Could not clean temp file {file_path}: {e}")
        
        # Clean up old backup files
        for user_dir in os.listdir(COLLECTIONS_DIR):
            user_path = os.path.join(COLLECTIONS_DIR, user_dir)
            if not os.path.isdir(user_path):
                continue
                
            for collection_dir in os.listdir(user_path):
                collection_path = os.path.join(user_path, collection_dir)
                if not os.path.isdir(collection_path):
                    continue
                
                for filename in os.listdir(collection_path):
                    if filename.endswith('.backup') or '.backup_' in filename:
                        backup_path = os.path.join(collection_path, filename)
                        try:
                            file_age = current_time - os.path.getctime(backup_path)
                            if file_age > 86400:  # 24 hours
                                file_size = os.path.getsize(backup_path) / (1024 * 1024)
                                os.remove(backup_path)
                                cleaned_resources["backup_files"] += 1
                                cleaned_resources["memory_freed_mb"] += file_size
                        except Exception as e:
                            logger.warning(f"Could not clean backup file {backup_path}: {e}")
        
        # Force garbage collection
        if PSUTIL_AVAILABLE:
            before_gc = psutil.Process().memory_info().rss / (1024 * 1024)
            
        gc.collect()
        
        if PSUTIL_AVAILABLE:
            after_gc = psutil.Process().memory_info().rss / (1024 * 1024)
            cleaned_resources["memory_freed_mb"] += max(0, before_gc - after_gc)
        
        logger.info(f"System cleanup completed: {cleaned_resources}")
        return cleaned_resources
        
    except Exception as e:
        logger.error(f"Error during system cleanup: {e}")
        return cleaned_resources

def get_processing_status(user_id, collection_id, memory_id):
    """Enhanced processing status with detailed metrics"""
    try:
        collection = get_collection(user_id, collection_id)
        if not collection:
            return None
        
        memory_metadata = None
        for memory in collection.get("memories", []):
            if memory["id"] == memory_id:
                memory_metadata = memory
                break
        
        if not memory_metadata:
            return None
        
        # Get chunk processing details
        chunks_db_path = get_collection_chunks_path(user_id, collection_id)
        chunk_details = {}
        
        if os.path.exists(chunks_db_path):
            with sqlite3.connect(chunks_db_path) as conn:
                cursor = conn.execute("""
                    SELECT embedding_status, COUNT(*) 
                    FROM chunks 
                    WHERE memory_id = ?
                    GROUP BY embedding_status
                """, (memory_id,))
                
                for status, count in cursor.fetchall():
                    chunk_details[status] = count
        
        return {
            "memory_id": memory_id,
            "status": memory_metadata.get("status", "unknown"),
            "basic_info": {
                "title": memory_metadata.get("title"),
                "type": memory_metadata.get("type"),
                "file_size_mb": memory_metadata.get("file_size_mb", 0),
                "text_length": memory_metadata.get("text_length", 0),
                "created_at": memory_metadata.get("created_at")
            },
            "processing_details": {
                "chunk_count": memory_metadata.get("chunk_count", 0),
                "successful_chunks": memory_metadata.get("successful_chunks", 0),
                "failed_chunks": memory_metadata.get("failed_chunks", 0),
                "embedding_success_rate": memory_metadata.get("embedding_success_rate", 0),
                "processing_time_seconds": memory_metadata.get("processing_time_seconds", 0)
            },
            "chunk_status": chunk_details,
            "searchable": chunk_details.get("completed", 0) > 0
        }
        
    except Exception as e:
        logger.error(f"Error getting processing status: {e}")
        return None

# Enhanced initialization and configuration
def initialize_system():
    """Enhanced system initialization with comprehensive setup"""
    try:
        # Clean up on startup
        cleanup_system_resources()
        
        # Validate system components
        validation_result = validate_system_integrity()
        
        # Log system status
        logger.info("Enhanced RAG System Initialization")
        logger.info("=" * 50)
        logger.info(f"Base Directory: {BASE_DIR}")
        logger.info(f"Collections Directory: {COLLECTIONS_DIR}")
        logger.info(f"Temporary Directory: {TEMP_DIR}")
        logger.info(f"Global Search Database: {GLOBAL_SEARCH_DB}")
        logger.info("")
        logger.info("Configuration:")
        logger.info(f"  Max File Size: {MAX_FILE_SIZE_MB}MB")
        logger.info(f"  Memory Limit: {MAX_MEMORY_USAGE_MB}MB")
        logger.info(f"  Chunk Size: {CHUNK_SIZE} characters")
        logger.info(f"  Chunk Overlap: {CHUNK_OVERLAP} characters")
        logger.info(f"  Batch Size: {BATCH_SIZE} embeddings")
        logger.info(f"  Embedding Dimension: {EMBEDDING_DIMENSION}")
        logger.info("")
        logger.info("Available Components:")
        logger.info(f"  Whisper Model: {'✓' if whisper_model else '✗'}")
        logger.info(f"  Memory Monitoring: {'✓' if PSUTIL_AVAILABLE else '✗'}")
        logger.info(f"  Embedding Generator: ✓")
        logger.info(f"  Global Search Manager: ✓")
        logger.info("")
        logger.info(f"System Health: {'✓ Healthy' if validation_result['healthy'] else '✗ Issues Found'}")
        
        if not validation_result['healthy']:
            logger.warning("System Issues:")
            for issue in validation_result['issues']:
                logger.warning(f"  - {issue}")
        
        logger.info("=" * 50)
        logger.info("System initialization completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        return False

# Initialize system on import
_system_initialized = initialize_system()

# Export validation for external health checks
def get_system_health():
    """Get current system health status"""
    return {
        "initialized": _system_initialized,
        "components": {
            "whisper_model": whisper_model is not None,
            "memory_monitoring": PSUTIL_AVAILABLE,
            "embedding_generator": True,
            "global_search_manager": True
        },
        "directories": {
            "base_dir_exists": os.path.exists(BASE_DIR),
            "collections_dir_exists": os.path.exists(COLLECTIONS_DIR),
            "temp_dir_exists": os.path.exists(TEMP_DIR)
        },
        "database": {
            "global_search_db_exists": os.path.exists(GLOBAL_SEARCH_DB)
        }
    }
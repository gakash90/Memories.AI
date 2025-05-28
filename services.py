import ollama
import os
import uuid
import json
import shutil
import mimetypes
from datetime import datetime
import numpy as np
import faiss
import whisper
import fitz
import sqlite3
from werkzeug.utils import secure_filename
import hashlib
import re
from rank_bm25 import BM25Okapi
import logging
import threading
import gc
import time

# Enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components with better models
whisper_model = whisper.load_model("base")  # Better accuracy than tiny

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
    'image': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}  # For OCR support
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
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring disabled")
# Global search index manager
class GlobalSearchManager:
    """Manages global search across all collections and users"""
    
    def __init__(self):
        self.db_path = GLOBAL_SEARCH_DB
        self.bm25_cache = {}
        self.tfidf_cache = {}
        self.lock = threading.Lock()
        self._init_global_db()
    
    def _init_global_db(self):
        """Initialize global search database"""
        with sqlite3.connect(self.db_path) as conn:
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
                    updated_at TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON global_documents(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_collection_id ON global_documents(collection_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON global_documents(content_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON global_documents(memory_type)")
            
            # Full-text search support
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    id, title, content, 
                    content='global_documents',
                    content_rowid='rowid'
                )
            """)
    
    def index_document(self, user_id, collection_id, memory_id, title, content, memory_type, metadata=None):
        """Add or update document in global search index"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert or update main table
            conn.execute("""
                INSERT OR REPLACE INTO global_documents 
                (id, user_id, collection_id, memory_id, title, content, content_hash, memory_type, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_id, user_id, collection_id, memory_id, title, content, 
                content_hash, memory_type, json.dumps(metadata or {}), 
                datetime.now().isoformat(), datetime.now().isoformat()
            ))
            
            # Update FTS index
            conn.execute("""
                INSERT OR REPLACE INTO documents_fts (id, title, content)
                VALUES (?, ?, ?)
            """, (memory_id, title, content))
        
        # Clear caches
        with self.lock:
            self.bm25_cache.clear()
            self.tfidf_cache.clear()
    
    def remove_document(self, memory_id):
        """Remove document from global search index"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM global_documents WHERE id = ?", (memory_id,))
            conn.execute("DELETE FROM documents_fts WHERE id = ?", (memory_id,))
        
        with self.lock:
            self.bm25_cache.clear()
            self.tfidf_cache.clear()

# Global instance
global_search_manager = GlobalSearchManager()

# Enhanced utility functions
def detect_file_type(file):
    """Enhanced file type detection with better accuracy"""
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
    """Enhanced file validation"""
    if not filename or '.' not in filename:
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

# REPLACE your existing chunk_text function with this enhanced version:

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Enhanced text chunking with smart boundaries"""
    if not text:
        return []
    
    text_length = len(text)
    if text_length <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < text_length:
        end = start + chunk_size
        
        if end < text_length:
            # Try multiple boundary types in order of preference
            boundaries = [
                ('\n\n', 300),  # Paragraph boundaries
                ('.', 200),     # Sentence boundaries  
                ('!', 200),     # Exclamation boundaries
                ('?', 200),     # Question boundaries
                ('\n', 100),    # Line boundaries
                (' ', 50)       # Word boundaries
            ]
            
            for boundary_char, search_range in boundaries:
                boundary_pos = text.rfind(boundary_char, max(start, end - search_range), end)
                if boundary_pos > start:
                    end = boundary_pos + len(boundary_char)
                    break
        
        chunk = text[start:end].strip()
        
        if len(chunk) >= MIN_CHUNK_SIZE or start + len(chunk) >= text_length:
            if chunk:
                chunks.append(chunk)
        
        start = end - overlap
        if start >= text_length:
            break
        
        if start <= end - chunk_size:
            start = end
    
    return chunks

# THEN UPDATE your process_memory function to use smart_chunk_text instead of chunk_text:
# FIND this line:
# REPLACE with:

# Enhanced path management functions (keeping original names)
def get_user_collections_dir(user_id):
    """Enhanced user directory management"""
    user_dir = os.path.join(COLLECTIONS_DIR, f'user_{user_id}')
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def get_collection_path(user_id, collection_id):
    """Enhanced collection path management"""
    collection_path = os.path.join(get_user_collections_dir(user_id), collection_id)
    os.makedirs(collection_path, exist_ok=True)
    return collection_path

def get_collection_metadata_path(user_id, collection_id):
    """Get metadata file path with backup support"""
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

# Enhanced collection management (keeping original function names)
def create_collection(user_id, name, description=""):
    """Enhanced collection creation with better indexing"""
    collection_id = str(uuid.uuid4())
    
    user_collections_dir = get_user_collections_dir(user_id)
    collection_path = get_collection_path(user_id, collection_id)
    
    # Create metadata with enhanced fields
    metadata = {
        "id": collection_id,
        "user_id": user_id,
        "name": name,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "memories": [],
        "total_documents": 0,
        "total_chunks": 0,
        "supported_search": ["semantic", "keyword", "hybrid"],
        "index_version": "2.0"
    }
    
    # Save metadata
    with open(get_collection_metadata_path(user_id, collection_id), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Initialize enhanced HNSW index for better performance
    index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 100
    faiss.write_index(index, get_collection_index_path(user_id, collection_id))
    
    # Initialize chunks database
    _init_chunks_database(user_id, collection_id)
    
    logger.info(f"Created collection {collection_id} for user {user_id}")
    return collection_id, metadata

def _init_chunks_database(user_id, collection_id):
    """Initialize chunks database for this collection"""
    chunks_db_path = get_collection_chunks_path(user_id, collection_id)
    
    with sqlite3.connect(chunks_db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                memory_id TEXT,
                chunk_index INTEGER,
                total_chunks INTEGER,
                content TEXT,
                content_hash TEXT,
                embedding_index INTEGER,
                created_at TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_id ON chunks(memory_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_index ON chunks(chunk_index)")

def get_all_collections(user_id):
    """Enhanced collection listing with statistics"""
    collections = []
    user_collections_dir = get_user_collections_dir(user_id)
    
    if not os.path.exists(user_collections_dir):
        return collections
    
    for collection_id in os.listdir(user_collections_dir):
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
    
    return sorted(collections, key=lambda x: x.get('updated_at', ''), reverse=True)

def _get_collection_stats(user_id, collection_id):
    """Get real-time collection statistics"""
    stats = {
        "total_memories": 0,
        "total_chunks": 0,
        "memory_types": {},
        "total_size_mb": 0
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
        
        # Get chunk count
        chunks_db_path = get_collection_chunks_path(user_id, collection_id)
        if os.path.exists(chunks_db_path):
            with sqlite3.connect(chunks_db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM chunks")
                stats["total_chunks"] = cursor.fetchone()[0]
        
        # Get directory size
        collection_path = get_collection_path(user_id, collection_id)
        if os.path.exists(collection_path):
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(collection_path)
                for filename in filenames
            )
            stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)
    
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
    
    return stats

def get_collection(user_id, collection_id):
    """Enhanced collection retrieval with validation"""
    metadata_path = get_collection_metadata_path(user_id, collection_id)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                collection_data = json.load(f)
            
            # Validate and upgrade old collections
            if not collection_data.get("index_version"):
                collection_data = _upgrade_collection(user_id, collection_id, collection_data)
            
            return collection_data
        except Exception as e:
            logger.error(f"Error loading collection {collection_id}: {e}")
    
    return None

def _upgrade_collection(user_id, collection_id, collection_data):
    """Upgrade old collection format to new enhanced format"""
    logger.info(f"Upgrading collection {collection_id} to new format")
    
    collection_data.update({
        "updated_at": datetime.now().isoformat(),
        "total_documents": len(collection_data.get("memories", [])),
        "total_chunks": 0,
        "supported_search": ["semantic", "keyword", "hybrid"],
        "index_version": "2.0"
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
            
            # Transfer vectors
            if old_index.ntotal > 0:
                vectors = old_index.reconstruct_n(0, old_index.ntotal)
                new_index.add(vectors)
            
            faiss.write_index(new_index, new_index_path)
            logger.info(f"Upgraded index for collection {collection_id}")
            
        except Exception as e:
            logger.error(f"Error upgrading index: {e}")
            # Create new empty index
            index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, 32)
            faiss.write_index(index, new_index_path)
    
    # Save upgraded metadata
    with open(get_collection_metadata_path(user_id, collection_id), 'w') as f:
        json.dump(collection_data, f, indent=2)
    
    return collection_data

def delete_collection(user_id, collection_id):
    """Enhanced collection deletion with cleanup"""
    try:
        # Remove from global search index
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

# Enhanced text extraction functions (keeping original names)
# REPLACE your existing extract_text_from_pdf function with this:

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file with enhanced handling"""
    try:
        file_size = os.path.getsize(pdf_path)
        logger.info(f"Processing PDF: {file_size / (1024*1024):.1f}MB")
        
        # Check file size
        size_ok, error = check_file_size(pdf_path, 'pdf')
        if not size_ok:
            logger.error(error)
            return ""
        
        doc = fitz.open(pdf_path)
        text_parts = []
        total_pages = len(doc)
        
        # Process pages in batches for large PDFs
        batch_size = 10 if file_size > 50*1024*1024 else 50
        
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            
            for page_num in range(batch_start, batch_end):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    if page_text.strip():
                        text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {e}")
                    continue
            
            # Memory management
            if not check_memory_usage():
                gc.collect()
        
        doc.close()
        return "\n\n".join(text_parts)
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_audio(audio_path):
    """Enhanced audio transcription with better accuracy"""
    try:
        result = whisper_model.transcribe(
            audio_path,
            language=None,  # Auto-detect
            word_timestamps=True
        )
        
        # Build transcript with timestamps for better context
        transcript_parts = []
        for segment in result.get("segments", []):
            start_time = segment.get("start", 0)
            text = segment.get("text", "").strip()
            if text:
                transcript_parts.append(f"[{start_time:.1f}s] {text}")
        
        return "\n".join(transcript_parts) if transcript_parts else result.get("text", "")
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return ""

def extract_text_from_text_file(file_path):
    """Enhanced text file reading with encoding detection"""
    encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            break
    
    return ""

# Enhanced memory processing (keeping original function name)
def process_memory(user_id, collection_id, file, memory_type, title, description=""):
    """Enhanced memory processing with chunking and global indexing"""
    collection = get_collection(user_id, collection_id)
    if not collection:
        return None, "Collection not found"

    file_path = None
    
    try:
        memory_id = str(uuid.uuid4())
        
        # Save original file to temp location
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        # Save to temp first
        file_path = os.path.join(TEMP_DIR, f"temp_{memory_id}.{file_ext}")
        file.save(file_path)
        
        # Check file size from temp location
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)

        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            # Clean up temp file
            if os.path.exists(file_path):
                os.remove(file_path)
            return None, f"File too large ({file_size_mb:.1f}MB). Maximum size is {MAX_FILE_SIZE_MB}MB"

        # Prepare final location
        saved_filename = f"{memory_id}.{file_ext}"
        memory_dir = get_collection_documents_path(user_id, collection_id)
        os.makedirs(memory_dir, exist_ok=True)  # Ensure directory exists
        
        final_file_path = os.path.join(memory_dir, saved_filename)
        
        # Move file from temp to final location
        shutil.move(file_path, final_file_path)
        file_path = final_file_path  # Update file_path to point to final location
        
        # Extract text from final location
        memory_text = ""
        processing_metadata = {}
        
        if memory_type == 'audio':
            memory_text = extract_text_from_audio(file_path)
            processing_metadata["transcription"] = True
        elif memory_type == 'pdf':
            memory_text = extract_text_from_pdf(file_path)
            processing_metadata["pdf_extraction"] = True
        elif memory_type == 'text':
            memory_text = extract_text_from_text_file(file_path)
        else:
            # Auto-detect and process
            detected_type = detect_file_type(file)
            if detected_type == 'audio':
                memory_text = extract_text_from_audio(file_path)
            elif detected_type == 'pdf':
                memory_text = extract_text_from_pdf(file_path)
            else:
                memory_text = extract_text_from_text_file(file_path)
        
        if not memory_text or len(memory_text.strip()) < 10:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            return None, "Could not extract meaningful text from file"
            
        # Create memory metadata
        memory_metadata = {
            "id": memory_id,
            "title": title,
            "description": description,
            "type": memory_type,
            "filename": saved_filename,
            "original_filename": filename,
            "created_at": datetime.now().isoformat(),
            "processing_metadata": processing_metadata,
            "text_length": len(memory_text),
            "chunk_count": 0
        }
        
        # Chunk the text for better retrieval
        text_chunks = chunk_text(memory_text)
        memory_metadata["chunk_count"] = len(text_chunks)
        
        # Generate embeddings for all chunks
        embeddings = []
        chunk_data = []

        for batch_start in range(0, len(text_chunks), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(text_chunks))
            batch_chunks = text_chunks[batch_start:batch_end]
            
            batch_embeddings = []
            batch_chunk_data = []
            
            for i, chunk in enumerate(batch_chunks):
                chunk_index = batch_start + i
                try:
                    response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
                    embedding = np.array(response["embedding"]).astype('float32')
                    batch_embeddings.append(embedding)
                    
                    batch_chunk_data.append({
                        "chunk_id": f"{memory_id}_chunk_{chunk_index}",
                        "memory_id": memory_id,
                        "chunk_index": chunk_index,
                        "total_chunks": len(text_chunks),
                        "content": chunk,
                        "content_hash": hashlib.md5(chunk.encode()).hexdigest()
                    })
                    
                except Exception as e:
                    logger.error(f"Error generating embedding for chunk {chunk_index}: {e}")
                    continue
            
            embeddings.extend(batch_embeddings)
            chunk_data.extend(batch_chunk_data)

            if not check_memory_usage():
                gc.collect()
        
        if not embeddings:
            return None, "Failed to generate embeddings"
        
        # Update FAISS index
        index = faiss.read_index(get_collection_index_path(user_id, collection_id))
        embeddings_array = np.array(embeddings)
        start_index = index.ntotal
        index.add(embeddings_array)
        faiss.write_index(index, get_collection_index_path(user_id, collection_id))
        
        # Store chunks in database
        chunks_db_path = get_collection_chunks_path(user_id, collection_id)
        with sqlite3.connect(chunks_db_path) as conn:
            for i, chunk_info in enumerate(chunk_data):
                conn.execute("""
                    INSERT INTO chunks 
                    (chunk_id, memory_id, chunk_index, total_chunks, content, content_hash, embedding_index, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk_info["chunk_id"],
                    chunk_info["memory_id"],
                    chunk_info["chunk_index"],
                    chunk_info["total_chunks"],
                    chunk_info["content"],
                    chunk_info["content_hash"],
                    start_index + i,
                    datetime.now().isoformat()
                ))
        
        # Save full text content
        text_path = os.path.join(memory_dir, f"{memory_id}.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(memory_text)
        
        # Update collection metadata
        collection["memories"].append(memory_metadata)
        collection["total_documents"] = len(collection["memories"])
        collection["total_chunks"] = collection.get("total_chunks", 0) + len(text_chunks)
        collection["updated_at"] = datetime.now().isoformat()
        
        with open(get_collection_metadata_path(user_id, collection_id), 'w') as f:
            json.dump(collection, f, indent=2)
        
        # Add to global search index
        global_search_manager.index_document(
            user_id, collection_id, memory_id, title, memory_text, memory_type, memory_metadata
        )
        
        logger.info(f"Processed memory {memory_id} with {len(text_chunks)} chunks")
        return memory_metadata, None
    
    except Exception as e:
        logger.error(f"Error processing memory: {e}")
        
        # Clean up file if it exists
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
                
        return None, str(e)

# Enhanced query functions (keeping original function names)
def query_collection(user_id, collection_id, query_text, top_k=20, search_type="hybrid"):
    """Enhanced collection querying with multiple search strategies"""
    collection = get_collection(user_id, collection_id)
    if not collection or not collection.get("memories"):
        return [], "Collection not found or empty"
    
    try:
        # Semantic search using embeddings
        semantic_results = _semantic_search(user_id, collection_id, query_text, top_k * 2)
        
        # Keyword search using BM25
        keyword_results = _keyword_search(user_id, collection_id, query_text, top_k * 2)
        
        # Combine results based on search type
        if search_type == "semantic":
            final_results = semantic_results[:top_k]
        elif search_type == "keyword":
            final_results = keyword_results[:top_k]
        else:  # hybrid
            final_results = _combine_search_results(semantic_results, keyword_results, top_k)
        
        # Get full content for results
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
                # Get chunk content or full content
                if result.get("content"):
                    content = result["content"]
                else:
                    text_path = os.path.join(memory_dir, f"{memory_id}.txt")
                    with open(text_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                relevant_memories.append({
                    "metadata": memory_metadata,
                    "content": content,
                    "distance": result.get("distance", 0.0),
                    "score": result.get("score", 0.0),
                    "chunk_id": chunk_id,
                    "search_type": result.get("search_type", "unknown")
                })
        
        return relevant_memories, None
    
    except Exception as e:
        logger.error(f"Error querying collection: {e}")
        return [], str(e)

def _semantic_search(user_id, collection_id, query_text, top_k):
    """Perform semantic search using embeddings"""
    try:
        # Generate query embedding
        response = ollama.embeddings(model="nomic-embed-text", prompt=query_text)
        query_embedding = np.array([response["embedding"]]).astype('float32')
        
        # Search FAISS index
        index = faiss.read_index(get_collection_index_path(user_id, collection_id))
        
        if index.ntotal == 0:
            return []
        
        k = min(top_k, index.ntotal)
        distances, indices = index.search(query_embedding, k)
        
        # Get chunk information
        chunks_db_path = get_collection_chunks_path(user_id, collection_id)
        results = []
        
        with sqlite3.connect(chunks_db_path) as conn:
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:
                    continue
                
                cursor = conn.execute("""
                    SELECT chunk_id, memory_id, chunk_index, content 
                    FROM chunks 
                    WHERE embedding_index = ?
                """, (int(idx),))
                
                row = cursor.fetchone()
                if row:
                    chunk_id, memory_id, chunk_index, content = row
                    results.append({
                        "chunk_id": chunk_id,
                        "memory_id": memory_id,
                        "chunk_index": chunk_index,
                        "content": content,
                        "distance": float(distance),
                        "score": 1.0 / (1.0 + float(distance)),  # Convert distance to score
                        "search_type": "semantic"
                    })
        
        return results
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return []

def _keyword_search(user_id, collection_id, query_text, top_k):
    """Perform keyword search using BM25"""
    try:
        chunks_db_path = get_collection_chunks_path(user_id, collection_id)
        
        with sqlite3.connect(chunks_db_path) as conn:
            cursor = conn.execute("SELECT chunk_id, memory_id, chunk_index, content FROM chunks")
            chunks = cursor.fetchall()
        
        if not chunks:
            return []
        
        # Prepare documents for BM25
        documents = []
        chunk_info = []
        
        for chunk_id, memory_id, chunk_index, content in chunks:
            # Simple tokenization
            tokens = re.findall(r'\b\w+\b', content.lower())
            documents.append(tokens)
            chunk_info.append({
                "chunk_id": chunk_id,
                "memory_id": memory_id,
                "chunk_index": chunk_index,
                "content": content
            })
        
        # Perform BM25 search
        bm25 = BM25Okapi(documents)
        query_tokens = re.findall(r'\b\w+\b', query_text.lower())
        scores = bm25.get_scores(query_tokens)
        
        # Get top results
        scored_results = []
        for i, score in enumerate(scores):
            if score > 0:
                chunk_data = chunk_info[i]
                chunk_data.update({
                    "score": float(score),
                    "distance": 1.0 / (1.0 + float(score)),
                    "search_type": "keyword"
                })
                scored_results.append(chunk_data)
        
        # Sort by score and return top k
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]
        
    except Exception as e:
        logger.error(f"Error in keyword search: {e}")
        return []

def _combine_search_results(semantic_results, keyword_results, top_k):
    """Combine semantic and keyword search results"""
    # Create a mapping of chunk_id to results
    semantic_map = {r["chunk_id"]: r for r in semantic_results}
    keyword_map = {r["chunk_id"]: r for r in keyword_results}
    
    # Get all unique chunk IDs
    all_chunk_ids = set(semantic_map.keys()) | set(keyword_map.keys())
    
    combined_results = []
    
    for chunk_id in all_chunk_ids:
        semantic_result = semantic_map.get(chunk_id)
        keyword_result = keyword_map.get(chunk_id)
        
        # Calculate hybrid score
        semantic_score = semantic_result["score"] if semantic_result else 0.0
        keyword_score = keyword_result["score"] if keyword_result else 0.0
        
        # Weighted combination (can be tuned)
        hybrid_score = 0.6 * semantic_score + 0.4 * keyword_score
        
        # Use the result with more information
        base_result = semantic_result or keyword_result
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

def query_across_collections(user_id, query_text, collection_ids=None, top_k=50, search_type="hybrid"):
    """Search across multiple collections or all user collections"""
    try:
        if collection_ids is None:
            # Search all user collections
            collections = get_all_collections(user_id)
            collection_ids = [c["id"] for c in collections]
        
        all_results = []
        
        for collection_id in collection_ids:
            try:
                results, error = query_collection(user_id, collection_id, query_text, top_k, search_type)
                if not error and results:
                    # Add collection info to results
                    for result in results:
                        result["source_collection_id"] = collection_id
                    all_results.extend(results)
            except Exception as e:
                logger.error(f"Error searching collection {collection_id}: {e}")
                continue
        
        # Sort all results by score
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:top_k], None
        
    except Exception as e:
        logger.error(f"Error in cross-collection search: {e}")
        return [], str(e)

def global_search(user_id, query_text, search_type="hybrid", limit=100):
    """Search across all user documents using global index"""
    try:
        results = []
        
        # Full-text search using SQLite FTS
        with sqlite3.connect(global_search_manager.db_path) as conn:
            if search_type in ["keyword", "hybrid"]:
                cursor = conn.execute("""
                    SELECT gd.id, gd.collection_id, gd.title, gd.content, gd.memory_type, gd.metadata,
                           bm25(documents_fts) as rank
                    FROM documents_fts 
                    JOIN global_documents gd ON documents_fts.id = gd.id
                    WHERE documents_fts MATCH ? AND gd.user_id = ?
                    ORDER BY rank
                    LIMIT ?
                """, (query_text, user_id, limit))
                
                keyword_results = cursor.fetchall()
                
                for row in keyword_results:
                    doc_id, coll_id, title, content, mem_type, metadata_str, rank = row
                    results.append({
                        "document_id": doc_id,
                        "collection_id": coll_id,
                        "title": title,
                        "content": content[:500] + "..." if len(content) > 500 else content,
                        "memory_type": mem_type,
                        "metadata": json.loads(metadata_str) if metadata_str else {},
                        "score": float(rank) if rank else 0.0,
                        "search_type": "keyword_global"
                    })
        
        # Semantic search if requested
        if search_type in ["semantic", "hybrid"] and results:
            # Get embeddings for query
            response = ollama.embeddings(model="nomic-embed-text", prompt=query_text)
            query_embedding = np.array([response["embedding"]]).astype('float32')
            
            # Search each collection's FAISS index
            for result in results[:20]:  # Limit semantic search to top keyword results
                try:
                    collection_id = result["collection_id"]
                    semantic_results, _ = query_collection(user_id, collection_id, query_text, 5, "semantic")
                    
                    if semantic_results:
                        # Find matching document
                        for sem_result in semantic_results:
                            if sem_result["metadata"]["id"] == result["document_id"]:
                                result["semantic_score"] = sem_result.get("score", 0.0)
                                if search_type == "hybrid":
                                    result["score"] = 0.6 * result["score"] + 0.4 * result["semantic_score"]
                                break
                        
                except Exception as e:
                    logger.error(f"Error in semantic search for collection {collection_id}: {e}")
                    continue
        
        # Sort by final score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit], None
        
    except Exception as e:
        logger.error(f"Error in global search: {e}")
        return [], str(e)

def generate_response(query, relevant_memories, conversation_history=None):
    """Enhanced response generation with better context handling"""
    if not relevant_memories:
        return "I don't have any relevant memories to answer your question. Try rephrasing your query or check if you have uploaded related documents."
    
    # Prepare context with improved structure
    context_parts = []
    source_info = []
    
    for i, memory in enumerate(relevant_memories[:5]):  # Limit to top 5 for context
        metadata = memory['metadata']
        content = memory['content']
        
        # Truncate very long content
        if len(content) > 1000:
            content = content[:1000] + "..."
        
        source_type = metadata.get('type', 'unknown')
        title = metadata.get('title', 'Untitled')
        filename = metadata.get('original_filename', 'Unknown file')
        
        context_parts.append(f"""
Source {i+1}: {title}
Type: {source_type} ({filename})
Relevance Score: {memory.get('score', 0):.3f}
Content: {content}
""")
        
        source_info.append({
            "title": title,
            "type": source_type,
            "filename": filename,
            "score": memory.get('score', 0)
        })
    
    context = "\n".join(context_parts)
    
    # Build enhanced prompt
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

Your response:"""

    try:
        output = ollama.generate(
            model="llama3",
            prompt=prompt,
            options={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        )
        
        response = output['response']
        
        # Add source information
        if len(relevant_memories) > 1:
            sources_text = "\n\nSources referenced:"
            for i, info in enumerate(source_info):
                sources_text += f"\n{i+1}. {info['title']} ({info['type']}) - Relevance: {info['score']:.2f}"
            response += sources_text
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"I encountered an error while processing your question. Please try again. (Error: {str(e)})"

def query_specific_memory(user_id, collection_id, memory_id, query_text):
    """Enhanced specific memory querying with chunk-level search"""
    collection = get_collection(user_id, collection_id)
    if not collection:
        return None, "Collection not found"
    
    # Find the memory metadata
    memory_metadata = None
    for mem in collection.get("memories", []):
        if mem["id"] == memory_id:
            memory_metadata = mem
            break
    
    if not memory_metadata:
        return None, "Memory not found"
    
    try:
        # Get all chunks for this memory
        chunks_db_path = get_collection_chunks_path(user_id, collection_id)
        
        with sqlite3.connect(chunks_db_path) as conn:
            cursor = conn.execute("""
                SELECT chunk_id, chunk_index, content 
                FROM chunks 
                WHERE memory_id = ?
                ORDER BY chunk_index
            """, (memory_id,))
            
            chunks = cursor.fetchall()
        
        if not chunks:
            # Fallback to full text
            memory_dir = get_collection_documents_path(user_id, collection_id)
            text_path = os.path.join(memory_dir, f"{memory_id}.txt")
            
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return [{
                "metadata": memory_metadata,
                "content": content,
                "distance": 0.0,
                "score": 1.0
            }], None
        
        # Perform semantic search within this memory's chunks
        response = ollama.embeddings(model="nomic-embed-text", prompt=query_text)
        query_embedding = np.array([response["embedding"]]).astype('float32')
        
        # Score chunks based on keyword matching and semantic similarity
        scored_chunks = []
        query_tokens = set(re.findall(r'\b\w+\b', query_text.lower()))
        
        for chunk_id, chunk_index, content in chunks:
            # Simple keyword scoring
            content_tokens = set(re.findall(r'\b\w+\b', content.lower()))
            keyword_overlap = len(query_tokens & content_tokens) / len(query_tokens) if query_tokens else 0
            
            scored_chunks.append({
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "content": content,
                "keyword_score": keyword_overlap
            })
        
        # Sort by keyword score and take top chunks
        scored_chunks.sort(key=lambda x: x["keyword_score"], reverse=True)
        top_chunks = scored_chunks[:3]  # Top 3 most relevant chunks
        
        # If no good keyword matches, include first few chunks
        if not top_chunks or max(c["keyword_score"] for c in top_chunks) < 0.1:
            top_chunks = scored_chunks[:2]
        
        # Combine chunk contents
        combined_content = "\n\n".join([
            f"[Chunk {c['chunk_index'] + 1}] {c['content']}"
            for c in top_chunks
        ])
        
        return [{
            "metadata": memory_metadata,
            "content": combined_content,
            "distance": 0.0,
            "score": 1.0,
            "chunks_used": len(top_chunks)
        }], None
        
    except Exception as e:
        logger.error(f"Error querying specific memory: {e}")
        return None, str(e)

def delete_memory(user_id, collection_id, memory_id):
    """Enhanced memory deletion with proper cleanup"""
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
        
        # Remove from global search index
        global_search_manager.remove_document(memory_id)
        
        # Remove chunks from database
        chunks_db_path = get_collection_chunks_path(user_id, collection_id)
        with sqlite3.connect(chunks_db_path) as conn:
            cursor = conn.execute("SELECT embedding_index FROM chunks WHERE memory_id = ?", (memory_id,))
            embedding_indices = [row[0] for row in cursor.fetchall()]
            conn.execute("DELETE FROM chunks WHERE memory_id = ?", (memory_id,))
        
        # Remove files
        memory_dir = get_collection_documents_path(user_id, collection_id)
        
        # Remove original file
        original_file = os.path.join(memory_dir, memory["filename"])
        if os.path.exists(original_file):
            os.remove(original_file)
        
        # Remove text file
        text_file = os.path.join(memory_dir, f"{memory_id}.txt")
        if os.path.exists(text_file):
            os.remove(text_file)
        
        # Update collection metadata
        collection["memories"].pop(memory_index)
        collection["total_documents"] = len(collection["memories"])
        collection["updated_at"] = datetime.now().isoformat()
        
        # Recalculate total chunks
        with sqlite3.connect(chunks_db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            collection["total_chunks"] = cursor.fetchone()[0]
        
        with open(get_collection_metadata_path(user_id, collection_id), 'w') as f:
            json.dump(collection, f, indent=2)
        
        # Rebuild FAISS index to remove deleted embeddings
        rebuild_collection_index(user_id, collection_id)
        
        logger.info(f"Deleted memory {memory_id} from collection {collection_id}")
        return True, None
        
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        return False, str(e)

def rebuild_collection_index(user_id, collection_id):
    """Enhanced index rebuilding with chunk support"""
    try:
        collection = get_collection(user_id, collection_id)
        if not collection:
            return False
        
        # Create new HNSW index
        index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 100
        
        # Get all chunks and regenerate embeddings
        chunks_db_path = get_collection_chunks_path(user_id, collection_id)
        
        with sqlite3.connect(chunks_db_path) as conn:
            cursor = conn.execute("""
                SELECT chunk_id, content 
                FROM chunks 
                ORDER BY memory_id, chunk_index
            """)
            chunks = cursor.fetchall()
            
            if chunks:
                embeddings = []
                embedding_index = 0
                
                for chunk_id, content in chunks:
                    try:
                        response = ollama.embeddings(model="nomic-embed-text", prompt=content)
                        embedding = np.array([response["embedding"]]).astype('float32')
                        embeddings.append(embedding[0])
                        
                        # Update embedding index in database
                        conn.execute("""
                            UPDATE chunks 
                            SET embedding_index = ? 
                            WHERE chunk_id = ?
                        """, (embedding_index, chunk_id))
                        
                        embedding_index += 1
                        
                    except Exception as e:
                        logger.error(f"Error generating embedding for chunk {chunk_id}: {e}")
                        continue
                
                if embeddings:
                    embeddings_array = np.array(embeddings)
                    index.add(embeddings_array)
        
        # Save rebuilt index
        faiss.write_index(index, get_collection_index_path(user_id, collection_id))
        
        logger.info(f"Rebuilt index for collection {collection_id} with {index.ntotal} vectors")
        return True
        
    except Exception as e:
        logger.error(f"Error rebuilding collection index: {e}")
        return False

# Additional utility functions for enhanced search
def get_collection_statistics(user_id, collection_id):
    """Get detailed statistics for a collection"""
    try:
        collection = get_collection(user_id, collection_id)
        if not collection:
            return None
        
        stats = {
            "total_memories": len(collection.get("memories", [])),
            "memory_types": {},
            "total_chunks": 0,
            "index_size": 0,
            "disk_usage_mb": 0
        }
        
        # Count memory types
        for memory in collection.get("memories", []):
            mem_type = memory.get("type", "unknown")
            stats["memory_types"][mem_type] = stats["memory_types"].get(mem_type, 0) + 1
        
        # Get chunk count
        chunks_db_path = get_collection_chunks_path(user_id, collection_id)
        if os.path.exists(chunks_db_path):
            with sqlite3.connect(chunks_db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM chunks")
                stats["total_chunks"] = cursor.fetchone()[0]
        
        # Get index size
        index_path = get_collection_index_path(user_id, collection_id)
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            stats["index_size"] = index.ntotal
        
        # Calculate disk usage
        collection_path = get_collection_path(user_id, collection_id)
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(collection_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        stats["disk_usage_mb"] = round(total_size / (1024 * 1024), 2)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting collection statistics: {e}")
        return None

def search_suggestions(user_id, partial_query, limit=10):
    """Get search suggestions based on partial query"""
    try:
        suggestions = []
        
        # Search in global index for matching titles and content
        with sqlite3.connect(global_search_manager.db_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT title, memory_type 
                FROM global_documents 
                WHERE user_id = ? AND (
                    title LIKE ? OR 
                    content LIKE ?
                )
                LIMIT ?
            """, (user_id, f"%{partial_query}%", f"%{partial_query}%", limit))
            
            for title, mem_type in cursor.fetchall():
                suggestions.append({
                    "text": title,
                    "type": "document_title",
                    "memory_type": mem_type
                })
        
        # Add common search patterns
        common_patterns = [
            f"What is {partial_query}",
            f"How to {partial_query}",
            f"When did {partial_query}",
            f"Where is {partial_query}"
        ]
        
        for pattern in common_patterns:
            if len(suggestions) < limit:
                suggestions.append({
                    "text": pattern,
                    "type": "query_pattern"
                })
        
        return suggestions[:limit]
        
    except Exception as e:
        logger.error(f"Error getting search suggestions: {e}")
        return []
    
# ADD THESE NEW FUNCTIONS to your original services.py

def check_memory_usage():
    """Check current memory usage"""
    if not PSUTIL_AVAILABLE:
        return True
    
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > MAX_MEMORY_USAGE_MB:
            logger.warning(f"High memory usage: {memory_mb:.1f}MB")
            gc.collect()
            
        return memory_mb < MAX_MEMORY_USAGE_MB
    except:
        return True

def check_file_size(file_path, file_type):
    """Check if file size is within limits"""
    file_size = os.path.getsize(file_path)
    limit = FILE_SIZE_LIMITS.get(file_type, MAX_FILE_SIZE_MB * 1024 * 1024)
    
    if file_size > limit:
        size_mb = file_size / (1024 * 1024)
        limit_mb = limit / (1024 * 1024)
        return False, f"File size ({size_mb:.1f}MB) exceeds limit ({limit_mb:.1f}MB) for {file_type} files"
    
    return True, None


def extract_text_from_large_pdf(self, pdf_path):
        """Enhanced PDF extraction with memory management for large files"""
        try:
            file_size = os.path.getsize(pdf_path)
            logger.info(f"Processing PDF: {file_size / (1024*1024):.1f}MB")
            
            # Check file size
            size_ok, error = self.check_file_size(pdf_path, 'pdf')
            if not size_ok:
                return "", {"error": error}
            
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            text_parts = []
            metadata = {
                "total_pages": total_pages,
                "has_images": False,
                "has_tables": False,
                "processing_method": "batch",
                "file_size_mb": file_size / (1024*1024)
            }
            
            # Process pages in batches to manage memory
            batch_size = 10 if file_size > 50*1024*1024 else 50  # Smaller batches for large files
            
            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                logger.info(f"Processing PDF pages {batch_start+1}-{batch_end}/{total_pages}")
                
                batch_text = []
                
                for page_num in range(batch_start, batch_end):
                    try:
                        page = doc[page_num]
                        page_text = page.get_text()
                        
                        if page_text.strip():
                            batch_text.append(f"[Page {page_num + 1}]\n{page_text}")
                        
                        # Check for images and tables
                        if page.get_images():
                            metadata["has_images"] = True
                        if "|" in page_text or "table" in page_text.lower():
                            metadata["has_tables"] = True
                            
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num + 1}: {e}")
                        continue
                
                text_parts.extend(batch_text)
                
                # Memory management
                if not self.check_memory_usage():
                    logger.warning("High memory usage detected, forcing garbage collection")
                    gc.collect()
            
            doc.close()
            
            full_text = "\n\n".join(text_parts)
            metadata["extracted_text_length"] = len(full_text)
            
            logger.info(f"PDF extraction complete: {len(full_text)} characters from {total_pages} pages")
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"Error extracting large PDF: {e}")
            return "", {"error": str(e)}



# import ollama
# import os
# import uuid
# import json
# import shutil
# import mimetypes
# import tempfile
# from datetime import datetime
# import numpy as np
# import faiss
# import whisper
# import fitz
# from werkzeug.utils import secure_filename
# import hashlib
# import re
# import logging
# import threading
# import time
# import gc
# import sqlite3

# # Enhanced logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Try to import psutil for memory monitoring (optional)
# try:
#     import psutil
#     PSUTIL_AVAILABLE = True
# except ImportError:
#     PSUTIL_AVAILABLE = False
#     logger.warning("psutil not available - memory monitoring disabled")

# # Try to import BM25 for keyword search (optional)
# try:
#     from rank_bm25 import BM25Okapi
#     BM25_AVAILABLE = True
# except ImportError:
#     BM25_AVAILABLE = False
#     logger.warning("rank_bm25 not available - keyword search disabled")

# # Initialize components
# whisper_model = whisper.load_model("tiny")

# # Constants
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# COLLECTIONS_DIR = os.path.join(BASE_DIR, 'collections')
# os.makedirs(COLLECTIONS_DIR, exist_ok=True)
# TEMP_DIR = os.path.join(BASE_DIR, 'temp')
# os.makedirs(TEMP_DIR, exist_ok=True)

# # Global search database
# GLOBAL_SEARCH_DB = os.path.join(BASE_DIR, 'global_search.db')

# # Embedding dimension for vector database
# EMBEDDING_DIMENSION = 768

# # Enhanced chunking configuration
# CHUNK_SIZE = 800
# CHUNK_OVERLAP = 150
# MAX_CHUNK_SIZE = 1200
# MIN_CHUNK_SIZE = 100

# # Large document handling
# MAX_FILE_SIZE_MB = 500
# MAX_MEMORY_USAGE_MB = 2048
# BATCH_SIZE = 50

# # Allowed file extensions
# ALLOWED_EXTENSIONS = {
#     'audio': {'wav', 'mp3', 'ogg', 'm4a', 'flac', 'aac'},
#     'pdf': {'pdf'},
#     'text': {'txt', 'md', 'csv', 'json', 'rtf', 'log'},
#     'document': {'doc', 'docx', 'odt'},
#     'image': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
# }

# # File size limits by type
# FILE_SIZE_LIMITS = {
#     'pdf': 200 * 1024 * 1024,      # 200MB for PDFs
#     'audio': 500 * 1024 * 1024,    # 500MB for audio
#     'text': 100 * 1024 * 1024,     # 100MB for text
#     'document': 50 * 1024 * 1024   # 50MB for docs
# }

# # Mapping of MIME types to memory types
# MIME_TYPE_MAP = {
#     'audio/mpeg': 'audio',
#     'audio/mp3': 'audio',
#     'audio/ogg': 'audio',
#     'audio/wav': 'audio',
#     'audio/x-wav': 'audio',
#     'audio/m4a': 'audio',
#     'audio/mp4': 'audio',
#     'audio/flac': 'audio',
#     'audio/aac': 'audio',
#     'application/pdf': 'pdf',
#     'text/plain': 'text',
#     'text/markdown': 'text',
#     'text/csv': 'text',
#     'application/json': 'text',
#     'application/rtf': 'text',
#     'application/msword': 'document',
#     'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'document',
#     'image/png': 'image',
#     'image/jpeg': 'image',
#     'image/gif': 'image'
# }

# # Global search manager
# class GlobalSearchManager:
#     """Manages global search across all collections and users"""
    
#     def __init__(self):
#         self.db_path = GLOBAL_SEARCH_DB
#         self.lock = threading.Lock()
#         self._init_global_db()
    
#     def _init_global_db(self):
#         """Initialize global search database"""
#         with sqlite3.connect(self.db_path) as conn:
#             conn.execute("""
#                 CREATE TABLE IF NOT EXISTS global_documents (
#                     id TEXT PRIMARY KEY,
#                     user_id INTEGER,
#                     collection_id TEXT,
#                     memory_id TEXT,
#                     title TEXT,
#                     content TEXT,
#                     content_hash TEXT,
#                     memory_type TEXT,
#                     metadata TEXT,
#                     created_at TEXT,
#                     updated_at TEXT
#                 )
#             """)
#             conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON global_documents(user_id)")
#             conn.execute("CREATE INDEX IF NOT EXISTS idx_collection_id ON global_documents(collection_id)")
#             conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON global_documents(content_hash)")
    
#     def index_document(self, user_id, collection_id, memory_id, title, content, memory_type, metadata=None):
#         """Add or update document in global search index"""
#         content_hash = hashlib.md5(content.encode()).hexdigest()
        
#         with sqlite3.connect(self.db_path) as conn:
#             conn.execute("""
#                 INSERT OR REPLACE INTO global_documents 
#                 (id, user_id, collection_id, memory_id, title, content, content_hash, memory_type, metadata, created_at, updated_at)
#                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#             """, (
#                 memory_id, user_id, collection_id, memory_id, title, content, 
#                 content_hash, memory_type, json.dumps(metadata or {}), 
#                 datetime.now().isoformat(), datetime.now().isoformat()
#             ))
    
#     def remove_document(self, memory_id):
#         """Remove document from global search index"""
#         with sqlite3.connect(self.db_path) as conn:
#             conn.execute("DELETE FROM global_documents WHERE id = ?", (memory_id,))

# # Global instance
# global_search_manager = GlobalSearchManager()

# # Utility Functions
# def detect_file_type(file):
#     """Automatically detect the file type based on extension and MIME type"""
#     filename = file.filename
#     # Check file extension first
#     ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
#     for type_key, extensions in ALLOWED_EXTENSIONS.items():
#         if ext in extensions:
#             return type_key
    
#     # If extension doesn't match, try MIME type
#     mime_type = mimetypes.guess_type(filename)[0]
#     if mime_type in MIME_TYPE_MAP:
#         return MIME_TYPE_MAP[mime_type]
    
#     # Default to text if we can't determine the type
#     return None

# def allowed_file(filename, file_type):
#     """Check if file is allowed"""
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS.get(file_type, set())

# def check_memory_usage():
#     """Check current memory usage"""
#     if not PSUTIL_AVAILABLE:
#         return True
    
#     try:
#         process = psutil.Process()
#         memory_mb = process.memory_info().rss / 1024 / 1024
        
#         if memory_mb > MAX_MEMORY_USAGE_MB:
#             logger.warning(f"High memory usage: {memory_mb:.1f}MB")
#             gc.collect()
            
#         return memory_mb < MAX_MEMORY_USAGE_MB
#     except:
#         return True

# def check_file_size(file_path, file_type):
#     """Check if file size is within limits"""
#     file_size = os.path.getsize(file_path)
#     limit = FILE_SIZE_LIMITS.get(file_type, MAX_FILE_SIZE_MB * 1024 * 1024)
    
#     if file_size > limit:
#         size_mb = file_size / (1024 * 1024)
#         limit_mb = limit / (1024 * 1024)
#         return False, f"File size ({size_mb:.1f}MB) exceeds limit ({limit_mb:.1f}MB) for {file_type} files"
    
#     return True, None

# def smart_chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
#     """Enhanced text chunking with smart boundaries"""
#     if not text:
#         return []
    
#     text_length = len(text)
#     if text_length <= chunk_size:
#         return [text]
    
#     chunks = []
#     start = 0
    
#     while start < text_length:
#         end = start + chunk_size
        
#         if end < text_length:
#             # Try multiple boundary types in order of preference
#             boundaries = [
#                 ('\n\n', 300),  # Paragraph boundaries
#                 ('.', 200),     # Sentence boundaries  
#                 ('!', 200),     # Exclamation boundaries
#                 ('?', 200),     # Question boundaries
#                 ('\n', 100),    # Line boundaries
#                 (' ', 50)       # Word boundaries
#             ]
            
#             for boundary_char, search_range in boundaries:
#                 boundary_pos = text.rfind(boundary_char, max(start, end - search_range), end)
#                 if boundary_pos > start:
#                     end = boundary_pos + len(boundary_char)
#                     break
        
#         chunk = text[start:end].strip()
        
#         if len(chunk) >= MIN_CHUNK_SIZE or start + len(chunk) >= text_length:
#             if chunk:
#                 chunks.append(chunk)
        
#         start = end - overlap
#         if start >= text_length:
#             break
        
#         if start <= end - chunk_size:
#             start = end
    
#     return chunks

# # Path management functions
# def get_user_collections_dir(user_id):
#     """Get the directory for a specific user's collections"""
#     return os.path.join(COLLECTIONS_DIR, f'user_{user_id}')

# def get_collection_path(user_id, collection_id):
#     """Get the path for a specific collection"""
#     return os.path.join(get_user_collections_dir(user_id), collection_id)

# def get_collection_metadata_path(user_id, collection_id):
#     """Get the metadata file path for a specific collection"""
#     return os.path.join(get_collection_path(user_id, collection_id), 'metadata.json')

# def get_collection_index_path(user_id, collection_id):
#     """Enhanced index path with backward compatibility"""
#     collection_path = get_collection_path(user_id, collection_id)
    
#     # Check for old index first (backward compatibility)
#     old_index_path = os.path.join(collection_path, 'index')
#     if os.path.exists(old_index_path):
#         return old_index_path
    
#     # Return new HNSW path for new collections
#     return os.path.join(collection_path, 'hnsw_index.faiss')

# def get_collection_documents_path(user_id, collection_id):
#     """Get the documents directory for a specific collection"""
#     return os.path.join(get_collection_path(user_id, collection_id), 'documents')

# def get_collection_chunks_path(user_id, collection_id):
#     """Get the chunks database path for this collection"""
#     return os.path.join(get_collection_path(user_id, collection_id), 'chunks.db')

# # Collection Management Functions
# def create_collection(user_id, name, description=""):
#     """Create a new collection with unique ID"""
#     collection_id = str(uuid.uuid4())
    
#     # Ensure user directory exists
#     user_collections_dir = get_user_collections_dir(user_id)
#     os.makedirs(user_collections_dir, exist_ok=True)
    
#     collection_path = get_collection_path(user_id, collection_id)
    
#     # Create collection directory structure
#     os.makedirs(collection_path, exist_ok=True)
#     os.makedirs(get_collection_documents_path(user_id, collection_id), exist_ok=True)
    
#     # Create metadata file
#     metadata = {
#         "id": collection_id,
#         "user_id": user_id,
#         "name": name,
#         "description": description,
#         "created_at": datetime.now().isoformat(),
#         "memories": []
#     }
    
#     with open(get_collection_metadata_path(user_id, collection_id), 'w') as f:
#         json.dump(metadata, f)
    
#     # Initialize FAISS index (try HNSW, fallback to flat)
#     try:
#         index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, 32)
#         index.hnsw.efConstruction = 200
#         index.hnsw.efSearch = 100
#     except:
#         # Fallback to flat index if HNSW not available
#         index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    
#     faiss.write_index(index, get_collection_index_path(user_id, collection_id))
    
#     # Initialize chunks database
#     _init_chunks_database(user_id, collection_id)
    
#     return collection_id, metadata

# def _init_chunks_database(user_id, collection_id):
#     """Initialize chunks database for this collection"""
#     chunks_db_path = get_collection_chunks_path(user_id, collection_id)
    
#     with sqlite3.connect(chunks_db_path) as conn:
#         conn.execute("""
#             CREATE TABLE IF NOT EXISTS chunks (
#                 chunk_id TEXT PRIMARY KEY,
#                 memory_id TEXT,
#                 chunk_index INTEGER,
#                 total_chunks INTEGER,
#                 content TEXT,
#                 content_hash TEXT,
#                 embedding_index INTEGER,
#                 created_at TEXT
#             )
#         """)
#         conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_id ON chunks(memory_id)")
#         conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_index ON chunks(chunk_index)")

# def get_all_collections(user_id):
#     """Get list of all collections for a specific user"""
#     collections = []
#     user_collections_dir = get_user_collections_dir(user_id)
    
#     if not os.path.exists(user_collections_dir):
#         return collections
    
#     for collection_id in os.listdir(user_collections_dir):
#         metadata_path = get_collection_metadata_path(user_id, collection_id)
#         if os.path.exists(metadata_path):
#             try:
#                 with open(metadata_path, 'r') as f:
#                     collections.append(json.load(f))
#             except Exception as e:
#                 logger.error(f"Error loading collection {collection_id}: {e}")
    
#     return collections

# def get_collection(user_id, collection_id):
#     """Get collection metadata by ID"""
#     metadata_path = get_collection_metadata_path(user_id, collection_id)
#     if os.path.exists(metadata_path):
#         try:
#             with open(metadata_path, 'r') as f:
#                 return json.load(f)
#         except Exception as e:
#             logger.error(f"Error loading collection {collection_id}: {e}")
#     return None

# def delete_collection(user_id, collection_id):
#     """Delete a collection and all its data"""
#     try:
#         # Remove from global search index
#         collection = get_collection(user_id, collection_id)
#         if collection:
#             for memory in collection.get("memories", []):
#                 global_search_manager.remove_document(memory["id"])
        
#         # Remove collection directory
#         collection_path = get_collection_path(user_id, collection_id)
#         if os.path.exists(collection_path):
#             shutil.rmtree(collection_path)
        
#         return True
#     except Exception as e:
#         logger.error(f"Error deleting collection: {e}")
#         return False

# # Memory Processing Functions
# def extract_text_from_pdf(pdf_path):
#     """Extract text content from a PDF file with enhanced handling"""
#     try:
#         file_size = os.path.getsize(pdf_path)
#         logger.info(f"Processing PDF: {file_size / (1024*1024):.1f}MB")
        
#         # Check file size
#         size_ok, error = check_file_size(pdf_path, 'pdf')
#         if not size_ok:
#             logger.error(error)
#             return ""
        
#         doc = fitz.open(pdf_path)
#         text_parts = []
#         total_pages = len(doc)
        
#         # Process pages in batches for large PDFs
#         batch_size = 10 if file_size > 50*1024*1024 else 50
        
#         for batch_start in range(0, total_pages, batch_size):
#             batch_end = min(batch_start + batch_size, total_pages)
            
#             for page_num in range(batch_start, batch_end):
#                 try:
#                     page = doc[page_num]
#                     page_text = page.get_text()
#                     if page_text.strip():
#                         text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
#                 except Exception as e:
#                     logger.warning(f"Error processing page {page_num + 1}: {e}")
#                     continue
            
#             # Memory management
#             if not check_memory_usage():
#                 gc.collect()
        
#         doc.close()
#         return "\n\n".join(text_parts)
        
#     except Exception as e:
#         logger.error(f"Error extracting text from PDF: {e}")
#         return ""

# def extract_text_from_audio(audio_path):
#     """Extract text from audio file with enhanced handling"""
#     try:
#         file_size = os.path.getsize(audio_path)
#         logger.info(f"Processing audio: {file_size / (1024*1024):.1f}MB")
        
#         # Check file size
#         size_ok, error = check_file_size(audio_path, 'audio')
#         if not size_ok:
#             logger.error(error)
#             return ""
        
#         # Enhanced transcription
#         result = whisper_model.transcribe(
#             audio_path,
#             language=None,
#             word_timestamps=True,
#             verbose=False
#         )
        
#         # Build transcript with timestamps
#         transcript_parts = []
#         for segment in result.get("segments", []):
#             start_time = segment.get("start", 0)
#             text = segment.get("text", "").strip()
#             if text:
#                 transcript_parts.append(f"[{start_time:.1f}s] {text}")
        
#         return "\n".join(transcript_parts) if transcript_parts else result.get("text", "")
        
#     except Exception as e:
#         logger.error(f"Error transcribing audio: {e}")
#         return ""

# def extract_text_from_text_file(file_path):
#     """Extract text from text file with enhanced encoding detection"""
#     try:
#         file_size = os.path.getsize(file_path)
#         logger.info(f"Processing text file: {file_size / (1024*1024):.1f}MB")
        
#         # Check file size
#         size_ok, error = check_file_size(file_path, 'text')
#         if not size_ok:
#             logger.error(error)
#             return ""
        
#         encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']
        
#         for encoding in encodings:
#             try:
#                 # For large files, read in chunks
#                 if file_size > 10 * 1024 * 1024:  # 10MB+
#                     content_parts = []
#                     chunk_size = 1024 * 1024  # 1MB chunks
                    
#                     with open(file_path, 'r', encoding=encoding) as f:
#                         while True:
#                             chunk = f.read(chunk_size)
#                             if not chunk:
#                                 break
#                             content_parts.append(chunk)
                            
#                             if not check_memory_usage():
#                                 gc.collect()
                    
#                     return ''.join(content_parts)
#                 else:
#                     with open(file_path, 'r', encoding=encoding) as f:
#                         return f.read()
                        
#             except (UnicodeDecodeError, UnicodeError):
#                 continue
#             except Exception as e:
#                 logger.error(f"Error reading text file with {encoding}: {e}")
#                 break
        
#         return ""
        
#     except Exception as e:
#         logger.error(f"Error reading text file: {e}")
#         return ""

# def process_memory(user_id, collection_id, file, memory_type, title, description=""):
#     """Process a new memory and add it to the collection with enhanced large file handling"""
#     collection = get_collection(user_id, collection_id)
#     if not collection:
#         return None, "Collection not found"
    
#     temp_file_path = None
    
#     try:
#         memory_id = str(uuid.uuid4())
        
#         # Save to temporary location first
#         filename = secure_filename(file.filename)
#         if not filename:
#             return None, "Invalid filename"
        
#         file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
#         temp_file_path = os.path.join(TEMP_DIR, f"temp_{memory_id}.{file_ext}")
        
#         file.save(temp_file_path)
        
#         # Check file size
#         file_size = os.path.getsize(temp_file_path)
#         file_size_mb = file_size / (1024 * 1024)
        
#         if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
#             return None, f"File too large ({file_size_mb:.1f}MB). Maximum size is {MAX_FILE_SIZE_MB}MB"
        
#         # Extract text based on memory type
#         memory_text = ""
        
#         start_time = time.time()
        
#         if memory_type == 'audio':
#             memory_text = extract_text_from_audio(temp_file_path)
#         elif memory_type == 'pdf':
#             memory_text = extract_text_from_pdf(temp_file_path)
#         elif memory_type == 'text':
#             memory_text = extract_text_from_text_file(temp_file_path)
#         else:
#             # Auto-detect and process
#             detected_type = detect_file_type(file)
#             if detected_type == 'audio':
#                 memory_text = extract_text_from_audio(temp_file_path)
#             elif detected_type == 'pdf':
#                 memory_text = extract_text_from_pdf(temp_file_path)
#             else:
#                 memory_text = extract_text_from_text_file(temp_file_path)
        
#         processing_time = time.time() - start_time
        
#         if not memory_text or len(memory_text.strip()) < 10:
#             return None, "Could not extract meaningful text from file"
        
#         # Move to permanent location
#         memory_dir = get_collection_documents_path(user_id, collection_id)
#         saved_filename = f"{memory_id}.{file_ext}"
#         final_file_path = os.path.join(memory_dir, saved_filename)
#         shutil.move(temp_file_path, final_file_path)
#         temp_file_path = None
        
#         # Create memory metadata
#         memory_metadata = {
#             "id": memory_id,
#             "title": title,
#             "description": description,
#             "type": memory_type,
#             "filename": saved_filename,
#             "original_filename": filename,
#             "created_at": datetime.now().isoformat(),
#             "text_length": len(memory_text),
#             "file_size_mb": file_size_mb,
#             "processing_time_seconds": processing_time,
#         }
        
#         # Enhanced chunking
#         text_chunks = smart_chunk_text(memory_text)
#         memory_metadata["chunk_count"] = len(text_chunks)
        
#         if len(text_chunks) > 1000:
#             return None, f"Document too complex: {len(text_chunks)} chunks (max 1000)"
        
#         # Generate embeddings in batches
#         all_embeddings = []
#         chunk_data = []
        
#         for batch_start in range(0, len(text_chunks), BATCH_SIZE):
#             batch_end = min(batch_start + BATCH_SIZE, len(text_chunks))
#             batch_chunks = text_chunks[batch_start:batch_end]
            
#             batch_embeddings = []
#             batch_chunk_data = []
            
#             for i, chunk in enumerate(batch_chunks):
#                 chunk_index = batch_start + i
                
#                 try:
#                     response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
#                     embedding = np.array(response["embedding"]).astype('float32')
#                     batch_embeddings.append(embedding)
                    
#                     batch_chunk_data.append({
#                         "chunk_id": f"{memory_id}_chunk_{chunk_index}",
#                         "memory_id": memory_id,
#                         "chunk_index": chunk_index,
#                         "total_chunks": len(text_chunks),
#                         "content": chunk,
#                         "content_hash": hashlib.md5(chunk.encode()).hexdigest()
#                     })
                    
#                 except Exception as e:
#                     logger.error(f"Error generating embedding for chunk {chunk_index}: {e}")
#                     continue
            
#             all_embeddings.extend(batch_embeddings)
#             chunk_data.extend(batch_chunk_data)
            
#             # Memory management
#             if not check_memory_usage():
#                 gc.collect()
        
#         if not all_embeddings:
#             return None, "Failed to generate any embeddings"
        
#         # Update FAISS index
#         index = faiss.read_index(get_collection_index_path(user_id, collection_id))
#         embeddings_array = np.array(all_embeddings)
#         start_index = index.ntotal
#         index.add(embeddings_array)
#         faiss.write_index(index, get_collection_index_path(user_id, collection_id))
        
#         # Store chunks in database
#         chunks_db_path = get_collection_chunks_path(user_id, collection_id)
#         with sqlite3.connect(chunks_db_path) as conn:
#             for i, chunk_info in enumerate(chunk_data):
#                 conn.execute("""
#                     INSERT INTO chunks 
#                     (chunk_id, memory_id, chunk_index, total_chunks, content, content_hash, embedding_index, created_at)
#                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#                 """, (
#                     chunk_info["chunk_id"],
#                     chunk_info["memory_id"],
#                     chunk_info["chunk_index"],
#                     chunk_info["total_chunks"],
#                     chunk_info["content"],
#                     chunk_info["content_hash"],
#                     start_index + i,
#                     datetime.now().isoformat()
#                 ))
        
#         # Save full text content
#         text_path = os.path.join(memory_dir, f"{memory_id}.txt")
#         with open(text_path, 'w', encoding='utf-8') as f:
#             f.write(memory_text)
        
#         # Update collection metadata
#         collection["memories"].append(memory_metadata)
#         with open(get_collection_metadata_path(user_id, collection_id), 'w') as f:
#             json.dump(collection, f)
        
#         # Add to global search index
#         global_search_manager.index_document(
#             user_id, collection_id, memory_id, title, memory_text, memory_type, memory_metadata
#         )
        
#         logger.info(f"Successfully processed memory {memory_id}")
#         logger.info(f"- Text length: {len(memory_text):,} characters")
#         logger.info(f"- Chunks created: {len(text_chunks)}")
#         logger.info(f"- Processing time: {processing_time:.2f}s")
        
#         return memory_metadata, None
    
#     except Exception as e:
#         logger.error(f"Error processing memory: {e}")
        
#         # Cleanup temp file
#         if temp_file_path and os.path.exists(temp_file_path):
#             try:
#                 os.remove(temp_file_path)
#             except:
#                 pass
        
#         gc.collect()
#         return None, f"Error processing memory: {str(e)}"

# # Chat and Query Functions
# def query_collection(user_id, collection_id, query_text, top_k=3):
#     """Query a collection with a question and get relevant memories"""
#     collection = get_collection(user_id, collection_id)
#     if not collection or not collection.get("memories"):
#         return [], "Collection not found or empty"
    
#     try:
#         # Generate embedding for the query
#         response = ollama.embeddings(model="nomic-embed-text", prompt=query_text)
#         query_embedding = np.array([response["embedding"]]).astype('float32')
        
#         # Load FAISS index
#         index = faiss.read_index(get_collection_index_path(user_id, collection_id))
        
#         # Get top k most similar memories
#         k = min(top_k, index.ntotal)
#         if k == 0:
#             return [], "No memories found"
            
#         distances, indices = index.search(query_embedding, k)
        
#         # Get chunk information from database
#         chunks_db_path = get_collection_chunks_path(user_id, collection_id)
#         relevant_memories = []
        
#         # Check if chunks database exists
#         if os.path.exists(chunks_db_path):
#             with sqlite3.connect(chunks_db_path) as conn:
#                 for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
#                     if idx == -1:
#                         continue
                    
#                     cursor = conn.execute("""
#                         SELECT chunk_id, memory_id, content 
#                         FROM chunks 
#                         WHERE embedding_index = ?
#                     """, (int(idx),))
                    
#                     row = cursor.fetchone()
#                     if row:
#                         chunk_id, memory_id, content = row
                        
#                         # Get memory metadata
#                         memory_metadata = None
#                         for mem in collection["memories"]:
#                             if mem["id"] == memory_id:
#                                 memory_metadata = mem
#                                 break
                        
#                         if memory_metadata:
#                             relevant_memories.append({
#                                 "metadata": memory_metadata,
#                                 "content": content,
#                                 "distance": float(distance)
#                             })
#         else:
#             # Fallback to original method for collections without chunks database
#             memory_dir = get_collection_documents_path(user_id, collection_id)
            
#             for i in range(k):
#                 memory_index = indices[0][i]
#                 if memory_index < len(collection["memories"]):
#                     memory_metadata = collection["memories"][memory_index]
                    
#                     # Get text content
#                     text_path = os.path.join(memory_dir, f"{memory_metadata['id']}.txt")
#                     if os.path.exists(text_path):
#                         with open(text_path, 'r', encoding='utf-8') as f:
#                             content = f.read()
                        
#                         relevant_memories.append({
#                             "metadata": memory_metadata,
#                             "content": content,
#                             "distance": float(distances[0][i])
#                         })
        
#         return relevant_memories, None
    
#     except Exception as e:
#         logger.error(f"Error querying collection: {e}")
#         return [], str(e)

# def generate_response(query, relevant_memories, conversation_history=None):
#     """Generate a response based on relevant memories and optional conversation history"""
#     if not relevant_memories:
#         return "I don't have any relevant memories to answer your question."
    
#     # Combine memory contents for context
#     context = ""
#     for memory in relevant_memories:
#         context += f"Memory: {memory['metadata']['title']} (originally '{memory['metadata'].get('original_filename', 'unknown')}', type: {memory['metadata']['type']})\n{memory['content']}\n\n"
    
#     try:
#         # Enhanced prompt with conversation history
#         conversation_section = f"Previous conversation:\n{conversation_history}\n" if conversation_history else ""
#         history_guideline = "- Consider the conversation history above when answering - if the question refers to something mentioned in previous messages, use that context." if conversation_history else ""
        
#         prompt = f"""
# You are a highly capable AI assistant specialized in reasoning over personal memory data to help users retrieve meaningful information.

# {conversation_section}Your task is to read, analyze, and reference the user's memory content stored across various formats (including PDF documents, audio transcripts, and plain text). You must extract relevant information from these memories to accurately and helpfully answer the user's question.

# Guidelines:
# 1. Carefully go through all memory entries provided in the "Memories" section below. They may include:
#     - Text files (personal notes, journal entries, etc.)
#     - PDF files (e.g., scanned documents, eBooks, research papers)
#     - Audio transcripts (e.g., voice memos, recorded meetings)

# 2. When generating your response:
#     - Directly reference the specific memory content you used.
#     - If the memory comes from a particular file type, explicitly mention it (e.g., "In the PDF document about your travel itinerary..." or "According to the audio transcript from your voice memo on April 3rd...").
#     - Be accurate, concise, and clear. Your goal is to help the user recall or interpret their personal data.
#     - If multiple memory sources are relevant, compare and synthesize them.
#     {history_guideline}

# 3. If no memory is relevant to the user's question, politely inform the user that no related information was found, but suggest possible next steps (e.g., asking a broader or different question).

# Memories:
# {context}

# User question:
# {query}

# Your response:
# """
        
#         output = ollama.generate(
#             model="llama3",
#             prompt=prompt
#         )
        
#         return output['response']
    
#     except Exception as e:
#         logger.error(f"Error generating response: {e}")
#         return f"I had trouble processing your question. Technical error: {str(e)}"

# def query_specific_memory(user_id, collection_id, memory_id, query_text):
#     """Query a specific memory with a question"""
#     collection = get_collection(user_id, collection_id)
#     if not collection:
#         return None, "Collection not found"
    
#     # Find the specific memory in the collection
#     memory_metadata = None
#     for mem in collection.get("memories", []):
#         if mem["id"] == memory_id:
#             memory_metadata = mem
#             break
    
#     if not memory_metadata:
#         return None, "Memory not found"
    
#     try:
#         # Get text content of the memory
#         memory_dir = get_collection_documents_path(user_id, collection_id)
#         text_path = os.path.join(memory_dir, f"{memory_metadata['id']}.txt")
        
#         with open(text_path, 'r', encoding='utf-8') as f:
#             content = f.read()
        
#         # Create the memory object with metadata and content
#         memory = {
#             "metadata": memory_metadata,
#             "content": content,
#             "distance": 0.0  # Low value to indicate high relevance
#         }
        
#         return [memory], None
    
#     except Exception as e:
#         logger.error(f"Error querying specific memory: {e}")
#         return None, str(e)

# def delete_memory(user_id, collection_id, memory_id):
#     """Delete a memory from a collection"""
#     try:
#         # Get the collection
#         collection = get_collection(user_id, collection_id)
#         if not collection:
#             return False, "Collection not found"
        
#         # Find the memory in the collection
#         memory_index = None
#         for i, memory in enumerate(collection.get("memories", [])):
#             if memory["id"] == memory_id:
#                 memory_index = i
#                 break
        
#         if memory_index is None:
#             return False, "Memory not found"
        
#         # Remove the memory from collection metadata
#         memory = collection["memories"].pop(memory_index)
        
#         # Remove from global search index
#         global_search_manager.remove_document(memory_id)
        
#         # Delete the memory files from disk
#         memory_dir = get_collection_documents_path(user_id, collection_id)
        
#         # Delete the original file
#         original_file = os.path.join(memory_dir, memory["filename"])
#         if os.path.exists(original_file):
#             os.remove(original_file)
        
#         # Delete the text content file
#         text_file = os.path.join(memory_dir, f"{memory_id}.txt")
#         if os.path.exists(text_file):
#             os.remove(text_file)
        
#         # Remove chunks from database
#         chunks_db_path = get_collection_chunks_path(user_id, collection_id)
#         if os.path.exists(chunks_db_path):
#             with sqlite3.connect(chunks_db_path) as conn:
#                 conn.execute("DELETE FROM chunks WHERE memory_id = ?", (memory_id,))
        
#         # Save the updated collection metadata
#         with open(get_collection_metadata_path(user_id, collection_id), 'w') as f:
#             json.dump(collection, f)
        
#         # Rebuild FAISS index
#         rebuild_collection_index(user_id, collection_id)
        
#         return True, None
#     except Exception as e:
#         logger.error(f"Error deleting memory: {e}")
#         return False, str(e)

# def rebuild_collection_index(user_id, collection_id):
#     """Rebuild the FAISS index for a collection"""
#     try:
#         collection = get_collection(user_id, collection_id)
#         if not collection:
#             return False
        
#         # Create a new empty index
#         try:
#             index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, 32)
#             index.hnsw.efConstruction = 200
#             index.hnsw.efSearch = 100
#         except:
#             index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        
#         # Check if we have chunks database
#         chunks_db_path = get_collection_chunks_path(user_id, collection_id)
        
#         if os.path.exists(chunks_db_path):
#             # Rebuild from chunks database
#             with sqlite3.connect(chunks_db_path) as conn:
#                 cursor = conn.execute("""
#                     SELECT chunk_id, content 
#                     FROM chunks 
#                     ORDER BY memory_id, chunk_index
#                 """)
#                 chunks = cursor.fetchall()
                
#                 if chunks:
#                     embeddings = []
#                     embedding_index = 0
                    
#                     for chunk_id, content in chunks:
#                         try:
#                             response = ollama.embeddings(model="nomic-embed-text", prompt=content)
#                             embedding = np.array([response["embedding"]]).astype('float32')
#                             embeddings.append(embedding[0])
                            
#                             # Update embedding index in database
#                             conn.execute("""
#                                 UPDATE chunks 
#                                 SET embedding_index = ? 
#                                 WHERE chunk_id = ?
#                             """, (embedding_index, chunk_id))
                            
#                             embedding_index += 1
                            
#                         except Exception as e:
#                             logger.error(f"Error generating embedding for chunk {chunk_id}: {e}")
#                             continue
                    
#                     if embeddings:
#                         embeddings_array = np.array(embeddings)
#                         index.add(embeddings_array)
#         else:
#             # Fallback: rebuild from memory files
#             memory_dir = get_collection_documents_path(user_id, collection_id)
            
#             for memory in collection.get("memories", []):
#                 text_path = os.path.join(memory_dir, f"{memory['id']}.txt")
                
#                 if os.path.exists(text_path):
#                     # Read the memory text content
#                     with open(text_path, 'r', encoding='utf-8') as f:
#                         memory_text = f.read()
                    
#                     # Generate embedding
#                     try:
#                         response = ollama.embeddings(model="nomic-embed-text", prompt=memory_text)
#                         embedding = response["embedding"]
                        
#                         # Add to index
#                         index.add(np.array([embedding]).astype('float32'))
#                     except Exception as e:
#                         logger.error(f"Error generating embedding for memory {memory['id']}: {e}")
#                         continue
        
#         # Save the updated index
#         faiss.write_index(index, get_collection_index_path(user_id, collection_id))
        
#         logger.info(f"Rebuilt index for collection {collection_id} with {index.ntotal} vectors")
#         return True
        
#     except Exception as e:
#         logger.error(f"Error rebuilding collection index: {e}")
#         return False

# # Enhanced search functions for cross-collection and global search
# def query_across_collections(user_id, query_text, collection_ids=None, top_k=50):
#     """Search across multiple collections or all user collections"""
#     try:
#         if collection_ids is None:
#             # Search all user collections
#             collections = get_all_collections(user_id)
#             collection_ids = [c["id"] for c in collections]
        
#         all_results = []
        
#         for collection_id in collection_ids:
#             try:
#                 results, error = query_collection(user_id, collection_id, query_text, top_k)
#                 if not error and results:
#                     # Add collection info to results
#                     for result in results:
#                         result["source_collection_id"] = collection_id
#                     all_results.extend(results)
#             except Exception as e:
#                 logger.error(f"Error searching collection {collection_id}: {e}")
#                 continue
        
#         # Sort all results by relevance (distance)
#         all_results.sort(key=lambda x: x.get("distance", float('inf')))
#         return all_results[:top_k], None
        
#     except Exception as e:
#         logger.error(f"Error in cross-collection search: {e}")
#         return [], str(e)

# def global_search(user_id, query_text, limit=100):
#     """Search across all user documents using global index"""
#     try:
#         results = []
        
#         # Search using global documents database
#         with sqlite3.connect(global_search_manager.db_path) as conn:
#             cursor = conn.execute("""
#                 SELECT id, collection_id, title, content, memory_type, metadata
#                 FROM global_documents 
#                 WHERE user_id = ? AND (
#                     title LIKE ? OR 
#                     content LIKE ?
#                 )
#                 ORDER BY 
#                     CASE 
#                         WHEN title LIKE ? THEN 1
#                         WHEN content LIKE ? THEN 2
#                         ELSE 3
#                     END
#                 LIMIT ?
#             """, (
#                 user_id, 
#                 f"%{query_text}%", f"%{query_text}%",
#                 f"%{query_text}%", f"%{query_text}%",
#                 limit
#             ))
            
#             for row in cursor.fetchall():
#                 doc_id, coll_id, title, content, mem_type, metadata_str = row
#                 results.append({
#                     "document_id": doc_id,
#                     "collection_id": coll_id,
#                     "title": title,
#                     "content": content[:500] + "..." if len(content) > 500 else content,
#                     "memory_type": mem_type,
#                     "metadata": json.loads(metadata_str) if metadata_str else {},
#                     "search_type": "global"
#                 })
        
#         return results, None
        
#     except Exception as e:
#         logger.error(f"Error in global search: {e}")
#         return [], str(e)

# def search_suggestions(user_id, partial_query, limit=10):
#     """Get search suggestions based on partial query"""
#     try:
#         suggestions = []
        
#         # Search in global index for matching titles
#         with sqlite3.connect(global_search_manager.db_path) as conn:
#             cursor = conn.execute("""
#                 SELECT DISTINCT title, memory_type 
#                 FROM global_documents 
#                 WHERE user_id = ? AND title LIKE ?
#                 LIMIT ?
#             """, (user_id, f"%{partial_query}%", limit))
            
#             for title, mem_type in cursor.fetchall():
#                 suggestions.append({
#                     "text": title,
#                     "type": "document_title",
#                     "memory_type": mem_type
#                 })
        
#         return suggestions
        
#     except Exception as e:
#         logger.error(f"Error getting search suggestions: {e}")
#         return []

# # Additional utility functions
# def get_processing_status(user_id, collection_id, memory_id):
#     """Get processing status for a memory"""
#     try:
#         collection = get_collection(user_id, collection_id)
#         if not collection:
#             return None
        
#         for memory in collection.get("memories", []):
#             if memory["id"] == memory_id:
#                 return {
#                     "status": "completed",
#                     "chunk_count": memory.get("chunk_count", 0),
#                     "file_size_mb": memory.get("file_size_mb", 0),
#                     "processing_time": memory.get("processing_time_seconds", 0),
#                     "text_length": memory.get("text_length", 0)
#                 }
        
#         return None
        
#     except Exception as e:
#         logger.error(f"Error getting processing status: {e}")
#         return None

# def estimate_processing_time(file_size_mb, file_type):
#     """Estimate processing time for a file"""
#     # Rough estimates based on file type and size
#     estimates = {
#         'pdf': file_size_mb * 2,      # 2 seconds per MB
#         'audio': file_size_mb * 10,   # 10 seconds per MB
#         'text': file_size_mb * 0.5    # 0.5 seconds per MB
#     }
    
#     base_time = estimates.get(file_type, file_size_mb)
    
#     # Add embedding time (roughly 1 second per 1000 characters)
#     estimated_chars = file_size_mb * 1000 * 2  # Rough estimate
#     embedding_time = estimated_chars / 1000
    
#     total_time = base_time + embedding_time
    
#     return {
#         "estimated_seconds": int(total_time),
#         "estimated_minutes": round(total_time / 60, 1)
#     }

# def validate_file_before_upload(file, memory_type):
#     """Validate file before processing to give early feedback"""
#     try:
#         # Check if file is empty
#         file.seek(0, 2)  # Seek to end
#         file_size = file.tell()
#         file.seek(0)  # Reset to beginning
        
#         if file_size == 0:
#             return False, "File is empty"
        
#         file_size_mb = file_size / (1024 * 1024)
        
#         # Check file size limits
#         limit = FILE_SIZE_LIMITS.get(memory_type, MAX_FILE_SIZE_MB * 1024 * 1024)
#         limit_mb = limit / (1024 * 1024)
        
#         if file_size > limit:
#             return False, f"File size ({file_size_mb:.1f}MB) exceeds limit ({limit_mb:.1f}MB) for {memory_type} files"
        
#         # Check file extension
#         filename = file.filename
#         if not allowed_file(filename, memory_type):
#             return False, f"File type not supported for {memory_type}. Supported extensions: {', '.join(ALLOWED_EXTENSIONS.get(memory_type, []))}"
        
#         return True, None
        
#     except Exception as e:
#         logger.error(f"Error validating file: {e}")
#         return False, f"Error validating file: {str(e)}"

# def get_collection_statistics(user_id, collection_id):
#     """Get detailed statistics for a collection"""
#     try:
#         collection = get_collection(user_id, collection_id)
#         if not collection:
#             return None
        
#         stats = {
#             "total_memories": len(collection.get("memories", [])),
#             "memory_types": {},
#             "total_chunks": 0,
#             "index_size": 0,
#             "disk_usage_mb": 0
#         }
        
#         # Count memory types
#         for memory in collection.get("memories", []):
#             mem_type = memory.get("type", "unknown")
#             stats["memory_types"][mem_type] = stats["memory_types"].get(mem_type, 0) + 1
        
#         # Get chunk count
#         chunks_db_path = get_collection_chunks_path(user_id, collection_id)
#         if os.path.exists(chunks_db_path):
#             with sqlite3.connect(chunks_db_path) as conn:
#                 cursor = conn.execute("SELECT COUNT(*) FROM chunks")
#                 stats["total_chunks"] = cursor.fetchone()[0]
        
#         # Get index size
#         index_path = get_collection_index_path(user_id, collection_id)
#         if os.path.exists(index_path):
#             try:
#                 index = faiss.read_index(index_path)
#                 stats["index_size"] = index.ntotal
#             except:
#                 stats["index_size"] = 0
        
#         # Calculate disk usage
#         collection_path = get_collection_path(user_id, collection_id)
#         total_size = 0
#         for dirpath, dirnames, filenames in os.walk(collection_path):
#             for filename in filenames:
#                 filepath = os.path.join(dirpath, filename)
#                 try:
#                     total_size += os.path.getsize(filepath)
#                 except:
#                     continue
#         stats["disk_usage_mb"] = round(total_size / (1024 * 1024), 2)
        
#         return stats
        
#     except Exception as e:
#         logger.error(f"Error getting collection statistics: {e}")
#         return None

# def cleanup_temp_files():
#     """Clean up old temporary files"""
#     try:
#         current_time = time.time()
#         cleaned_count = 0
        
#         for filename in os.listdir(TEMP_DIR):
#             file_path = os.path.join(TEMP_DIR, filename)
            
#             # Remove files older than 1 hour
#             if os.path.isfile(file_path):
#                 try:
#                     file_age = current_time - os.path.getctime(file_path)
#                     if file_age > 3600:  # 1 hour in seconds
#                         os.remove(file_path)
#                         cleaned_count += 1
#                 except:
#                     pass
        
#         if cleaned_count > 0:
#             logger.info(f"Cleaned up {cleaned_count} temporary files")
        
#         return cleaned_count
        
#     except Exception as e:
#         logger.error(f"Error cleaning temp files: {e}")
#         return 0

# # System initialization
# def _initialize_system():
#     """Initialize the enhanced system"""
#     try:
#         # Clean up old temp files on startup
#         cleanup_temp_files()
        
#         # Log system initialization
#         logger.info("Enhanced RAG system initialized successfully")
#         logger.info(f"- Max file size: {MAX_FILE_SIZE_MB}MB")
#         logger.info(f"- Memory limit: {MAX_MEMORY_USAGE_MB}MB")
#         logger.info(f"- Chunk size: {CHUNK_SIZE} characters")
#         logger.info(f"- Batch size: {BATCH_SIZE} embeddings")
#         logger.info(f"- psutil available: {PSUTIL_AVAILABLE}")
#         logger.info(f"- BM25 available: {BM25_AVAILABLE}")
        
#     except Exception as e:
#         logger.error(f"Error initializing system: {e}")

# # Initialize on import
# _initialize_system()
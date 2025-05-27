import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///memory_vault.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    COLLECTIONS_DIR = os.path.join(BASE_DIR, 'collections')
    TEMP_DIR = os.path.join(BASE_DIR, 'temp')
    
    # Embedding dimension for the vector database
    EMBEDDING_DIMENSION = 768
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {
        'audio': {'wav', 'mp3', 'ogg', 'm4a'},
        'pdf': {'pdf'},
        'text': {'txt', 'md', 'csv', 'json'}
    }
    
    # Embedding model
    EMBEDDING_MODEL = "nomic-embed-text"
    
    # LLM model
    LLM_MODEL = "llama3"
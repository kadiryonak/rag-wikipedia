import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATA_DIR = os.getenv("DATA_DIR", "data")
    CHUNK_DIR = os.getenv("CHUNK_DIR", "chunks")
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", ***))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "wiki_collection")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

settings = Settings()


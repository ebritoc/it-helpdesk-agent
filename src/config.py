"""Configuration settings for the IT Helpdesk Ticket Assistance System"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Data file paths
NEW_TICKETS_PATH = DATA_DIR / "new_tickets.csv"
OLD_TICKETS_DIR = DATA_DIR / "old_tickets"

# HuggingFace API Configuration
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("HF_API_TOKEN not found in environment variables. Please set it in .env file")

# Model configurations
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# API Endpoints
# Router doesn't support embeddings - use original API with task-specific endpoints
EMBEDDING_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
LLM_API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"

# Retrieval parameters
TOP_K_RESULTS = 3  # Number of tickets showed for recommendation
SIMILARITY_THRESHOLD = 0.0  # Minimum similarity score (0-1)
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# LLM parameters
LLM_TEMPERATURE = 0.7
LLM_MAX_NEW_TOKENS = 500

# Cache file paths
EMBEDDINGS_CACHE_PATH = OUTPUTS_DIR / "embeddings_cache.pkl"
VECTOR_INDEX_PATH = OUTPUTS_DIR / "vector_index.pkl"

# Output file paths
RECOMMENDATIONS_JSON_PATH = OUTPUTS_DIR / "recommendations.json"
RECOMMENDATIONS_TXT_PATH = OUTPUTS_DIR / "recommendations.txt"

# API retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
BACKOFF_FACTOR = 2  # exponential backoff multiplier

# Qdrant Vector Store Configuration
QDRANT_STORAGE_PATH = OUTPUTS_DIR / "qdrant_storage"
QDRANT_COLLECTION_NAME = "helpdesk_tickets"
QDRANT_VECTOR_SIZE = 384  # Matches all-MiniLM-L6-v2
QDRANT_DISTANCE_METRIC = "Cosine"

# Hybrid Search Configuration
ENABLE_SPARSE_VECTORS = True  # Enable BM25 sparse vectors with RRF fusion
# Note: Qdrant's native BM25 uses server-side tokenization and IDF calculation
# Default BM25 parameters: k1=1.2, b=0.75 (not configurable in native mode)
# RRF (Reciprocal Rank Fusion) automatically balances dense + sparse results

# Metadata Filtering Configuration
ENABLE_CATEGORY_FILTER = True  # Enable category-based filtering

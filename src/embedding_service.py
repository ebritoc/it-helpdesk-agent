"""Embedding service for generating vector representations using HuggingFace API"""
import time
import numpy as np
import pickle
from typing import List
from pathlib import Path
from huggingface_hub import InferenceClient
from src.config import (
    HF_API_TOKEN,
    EMBEDDING_MODEL,
    MAX_RETRIES,
    RETRY_DELAY,
    BACKOFF_FACTOR,
    EMBEDDINGS_CACHE_PATH
)


class EmbeddingService:
    """Handles embedding generation via HuggingFace Inference API"""

    def __init__(self, api_token: str = None, model_name: str = None):
        """
        Initialize embedding service

        Args:
            api_token: HuggingFace API token (defaults to config)
            model_name: Model name (defaults to config)
        """
        self.api_token = api_token or HF_API_TOKEN
        self.model_name = model_name or EMBEDDING_MODEL
        self.client = InferenceClient(token=self.api_token)
        self.cache = {}

    def _call_api(self, text: str) -> np.ndarray:
        """
        Make API call to generate embedding with retry logic

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array

        Raises:
            Exception: If API call fails after retries
        """
        for attempt in range(MAX_RETRIES):
            try:
                # Use InferenceClient's feature_extraction method
                result = self.client.feature_extraction(text, model=self.model_name)

                # Convert to numpy array
                if isinstance(result, np.ndarray):
                    return result.astype(np.float32)
                elif isinstance(result, list):
                    return np.array(result, dtype=np.float32)
                else:
                    raise ValueError(f"Unexpected API response format: {type(result)}")

            except Exception as e:
                error_msg = str(e)
                print(f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {error_msg}")

                # Check if it's a model loading error
                if "503" in error_msg or "loading" in error_msg.lower():
                    wait_time = RETRY_DELAY * (attempt + 2)
                    print(f"Model loading, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue

                # Other errors: retry with exponential backoff
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (BACKOFF_FACTOR ** attempt)
                    time.sleep(wait_time)
                    continue

                # Last attempt failed
                raise

        raise Exception(f"Failed to generate embedding after {MAX_RETRIES} attempts")

    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings

        Returns:
            Embedding vector as numpy array
        """
        # Check cache first
        if use_cache and text in self.cache:
            return self.cache[text]

        # Generate embedding via API
        embedding = self._call_api(text)

        # Store in cache
        self.cache[text] = embedding

        return embedding

    def generate_embeddings_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed
            use_cache: Whether to use cached embeddings
            show_progress: Whether to print progress

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i, text in enumerate(texts):
            if show_progress:
                print(f"Generating embedding {i + 1}/{len(texts)}...", end='\r')

            embedding = self.generate_embedding(text, use_cache=use_cache)
            embeddings.append(embedding)

            # Small delay to avoid rate limiting
            if i < len(texts) - 1:
                time.sleep(0.5)

        if show_progress:
            print(f"Generated {len(embeddings)} embeddings successfully")

        return embeddings

    def save_cache(self, cache_path: Path = None):
        """
        Save embedding cache to disk

        Args:
            cache_path: Path to save cache file (defaults to config path)
        """
        cache_path = cache_path or EMBEDDINGS_CACHE_PATH

        with open(cache_path, 'wb') as f:
            pickle.dump(self.cache, f)

        print(f"Saved embedding cache to {cache_path}")

    def load_cache(self, cache_path: Path = None):
        """
        Load embedding cache from disk

        Args:
            cache_path: Path to cache file (defaults to config path)
        """
        cache_path = cache_path or EMBEDDINGS_CACHE_PATH

        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                self.cache = pickle.load(f)
            print(f"Loaded {len(self.cache)} cached embeddings from {cache_path}")
        else:
            print(f"No cache file found at {cache_path}")

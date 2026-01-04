"""Vector store for similarity search using cosine similarity"""
import numpy as np
import pickle
from typing import List, Dict, Any, Tuple
from pathlib import Path
from src.config import VECTOR_INDEX_PATH, TOP_K_RESULTS, SIMILARITY_THRESHOLD


class VectorStore:
    """In-memory vector store with cosine similarity search"""

    def __init__(self):
        """Initialize empty vector store"""
        self.embeddings = []  # List of numpy arrays
        self.tickets = []  # List of ticket metadata dictionaries
        self.embedding_matrix = None  # Stacked numpy array for efficient search

    def add_tickets(self, tickets: List[Dict[str, Any]], embeddings: List[np.ndarray]):
        """
        Add tickets and their embeddings to the store

        Args:
            tickets: List of ticket dictionaries with metadata
            embeddings: List of embedding vectors (numpy arrays)
        """
        if len(tickets) != len(embeddings):
            raise ValueError("Number of tickets and embeddings must match")

        self.tickets.extend(tickets)
        self.embeddings.extend(embeddings)

        # Stack embeddings into a matrix for efficient similarity computation
        self.embedding_matrix = np.vstack(self.embeddings)

        print(f"Added {len(tickets)} tickets to vector store (total: {len(self.tickets)})")

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)

        # Compute dot product
        similarity = np.dot(vec1_norm, vec2_norm)

        return float(similarity)

    def compute_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities between query and all stored embeddings

        Args:
            query_embedding: Query embedding vector

        Returns:
            Array of similarity scores
        """
        if self.embedding_matrix is None or len(self.embedding_matrix) == 0:
            return np.array([])

        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Normalize all stored embeddings
        norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True) + 1e-8
        embeddings_norm = self.embedding_matrix / norms

        # Compute similarities via matrix multiplication
        similarities = np.dot(embeddings_norm, query_norm)

        return similarities

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Search for most similar tickets

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return (defaults to config)
            threshold: Minimum similarity threshold (defaults to config)

        Returns:
            List of dictionaries with ticket data and similarity scores
        """
        top_k = top_k or TOP_K_RESULTS
        threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD

        if len(self.tickets) == 0:
            return []

        # Compute all similarities
        similarities = self.compute_similarities(query_embedding)

        # Create results with metadata
        results = []
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                result = {
                    'ticket': self.tickets[i].copy(),
                    'similarity_score': float(similarity)
                }
                results.append(result)

        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        # Return top-k results
        return results[:top_k]

    def save_index(self, index_path: Path = None):
        """
        Save vector store to disk

        Args:
            index_path: Path to save index file (defaults to config path)
        """
        index_path = index_path or VECTOR_INDEX_PATH

        data = {
            'tickets': self.tickets,
            'embeddings': self.embeddings,
            'embedding_matrix': self.embedding_matrix
        }

        with open(index_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved vector index to {index_path}")

    def load_index(self, index_path: Path = None):
        """
        Load vector store from disk

        Args:
            index_path: Path to index file (defaults to config path)
        """
        index_path = index_path or VECTOR_INDEX_PATH

        if not index_path.exists():
            print(f"No index file found at {index_path}")
            return False

        with open(index_path, 'rb') as f:
            data = pickle.load(f)

        self.tickets = data['tickets']
        self.embeddings = data['embeddings']
        self.embedding_matrix = data['embedding_matrix']

        print(f"Loaded vector index with {len(self.tickets)} tickets from {index_path}")
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        if len(self.tickets) == 0:
            return {'total_tickets': 0}

        categories = {}
        for ticket in self.tickets:
            category = ticket.get('category', 'Unknown')
            categories[category] = categories.get(category, 0) + 1

        return {
            'total_tickets': len(self.tickets),
            'embedding_dimension': self.embedding_matrix.shape[1] if self.embedding_matrix is not None else 0,
            'categories': categories
        }

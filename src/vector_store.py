"""Vector store for similarity search using Qdrant with hybrid search"""
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, SparseVector,
    SparseVectorParams, SparseIndexParams, Filter, FieldCondition, MatchValue,
    Prefetch, FusionQuery, Fusion
)

from src.config import (
    QDRANT_STORAGE_PATH, QDRANT_COLLECTION_NAME, QDRANT_VECTOR_SIZE,
    TOP_K_RESULTS, SIMILARITY_THRESHOLD, ENABLE_SPARSE_VECTORS,
    BM25_K1, BM25_B
)
from src.sparse_encoder import BM25SparseEncoder


class VectorStore:
    """Qdrant-based vector store with hybrid search and metadata filtering"""

    def __init__(self):
        """Initialize Qdrant client and sparse encoder"""
        # Initialize Qdrant with persistent storage
        self.client = QdrantClient(path=str(QDRANT_STORAGE_PATH))
        self.collection_name = QDRANT_COLLECTION_NAME

        # Initialize BM25 sparse encoder
        self.sparse_encoder = BM25SparseEncoder(k1=BM25_K1, b=BM25_B)

        # Track if collection is created
        self.collection_exists = False

        # Sparse encoder path (saved alongside Qdrant storage)
        self.encoder_path = QDRANT_STORAGE_PATH / "sparse_encoder.pkl"

    def _create_collection(self):
        """Create Qdrant collection with dense and sparse vector configurations"""
        # Check if collection already exists
        collections = self.client.get_collections().collections
        if any(col.name == self.collection_name for col in collections):
            print(f"[VectorStore] Collection '{self.collection_name}' already exists")
            self.collection_exists = True
            return

        # Define vector configurations
        vectors_config = {
            "dense": VectorParams(
                size=QDRANT_VECTOR_SIZE,
                distance=Distance.COSINE
            )
        }

        # Add sparse vector config if enabled
        sparse_vectors_config = None
        if ENABLE_SPARSE_VECTORS:
            sparse_vectors_config = {
                "sparse": SparseVectorParams(
                    index=SparseIndexParams()
                )
            }

        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config
        )

        # Create payload index for category filtering
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="category",
            field_schema="keyword"
        )

        print(f"[VectorStore] Created collection '{self.collection_name}'")
        self.collection_exists = True

    def add_tickets(self, tickets: List[Dict[str, Any]], embeddings: List[np.ndarray]):
        """Add tickets and their embeddings to the store

        Args:
            tickets: List of ticket dictionaries with metadata
            embeddings: List of dense embedding vectors (numpy arrays)
        """
        if len(tickets) != len(embeddings):
            raise ValueError("Number of tickets and embeddings must match")

        if not tickets:
            return

        # Create collection if it doesn't exist
        if not self.collection_exists:
            self._create_collection()

        # Prepare document texts for sparse encoder fitting
        documents = []
        for ticket in tickets:
            # Combine issue and description for BM25
            text = f"{ticket.get('issue', '')} {ticket.get('description', '')}"
            documents.append(text)

        # Fit sparse encoder on the corpus
        if ENABLE_SPARSE_VECTORS:
            print("[VectorStore] Fitting BM25 sparse encoder...")
            self.sparse_encoder.fit(documents)

        # Prepare points for batch upsert
        points = []
        for idx, (ticket, embedding, doc_text) in enumerate(zip(tickets, embeddings, documents)):
            # Prepare dense vector
            dense_vector = embedding.tolist()

            # Prepare sparse vector if enabled
            sparse_vector = None
            if ENABLE_SPARSE_VECTORS:
                sparse_data = self.sparse_encoder.encode(doc_text)
                sparse_vector = SparseVector(
                    indices=sparse_data["indices"],
                    values=sparse_data["values"]
                )

            # Create point with full ticket as payload
            point = PointStruct(
                id=idx,
                vector={
                    "dense": dense_vector,
                    **({"sparse": sparse_vector} if sparse_vector else {})
                },
                payload=ticket  # Store entire ticket as payload
            )
            points.append(point)

        # Batch upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"[VectorStore] Added {len(tickets)} tickets to Qdrant collection")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        threshold: float = None,
        category_filter: Optional[list] = None,  # Changed to list for multiple categories
        query_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for most similar tickets using hybrid search

        Args:
            query_embedding: Query dense embedding vector
            top_k: Number of results to return (defaults to config)
            threshold: Minimum similarity threshold (defaults to config)
            category_filter: Optional list of categories to filter by (e.g., ["Network", "Software"])
            query_text: Optional query text for sparse vector generation (enables hybrid search)

        Returns:
            List of dictionaries with ticket data and similarity scores
            Format: [{'ticket': {...}, 'similarity_score': float}, ...]
        """
        top_k = top_k or TOP_K_RESULTS
        threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD

        if not self.collection_exists:
            return []

        # Prepare category filter if specified (supports multiple categories)
        query_filter = None
        if category_filter and len(category_filter) > 0:
            from qdrant_client.models import MatchAny
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchAny(any=category_filter)
                    )
                ]
            )

        # If hybrid search is enabled and query_text is provided
        if ENABLE_SPARSE_VECTORS and query_text and self.sparse_encoder.is_fitted:
            # Generate sparse vector from query text
            sparse_data = self.sparse_encoder.encode(query_text)
            sparse_vector = SparseVector(
                indices=sparse_data["indices"],
                values=sparse_data["values"]
            )

            # Use Qdrant's built-in RRF fusion for proper hybrid search
            query_response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    # Dense vector search
                    Prefetch(
                        query=query_embedding.tolist(),
                        using="dense",
                        limit=top_k * 2  # Get more candidates for fusion
                    ),
                    # Sparse vector search (BM25)
                    Prefetch(
                        query=sparse_vector,
                        using="sparse",
                        limit=top_k * 2  # Get more candidates for fusion
                    ),
                ],
                # RRF fusion combines rankings from both searches
                query=FusionQuery(fusion=Fusion.RRF),
                query_filter=query_filter,  # Filter applied to final RRF results
                limit=top_k,
                score_threshold=threshold,
                with_payload=True
            )

            # Convert results to expected format
            results = []
            for point in query_response.points:
                result = {
                    'ticket': dict(point.payload),
                    'similarity_score': float(point.score)
                }
                results.append(result)

        else:
            # Dense-only search
            query_response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                using="dense",
                query_filter=query_filter,
                limit=top_k,
                score_threshold=threshold,
                with_payload=True
            )

            # Convert results to expected format
            results = []
            for point in query_response.points:
                result = {
                    'ticket': dict(point.payload),
                    'similarity_score': float(point.score)
                }
                results.append(result)

        return results[:top_k]

    def save_index(self, index_path: Path = None):
        """Save sparse encoder state (Qdrant auto-persists)

        Args:
            index_path: Not used (kept for API compatibility)
        """
        if ENABLE_SPARSE_VECTORS and self.sparse_encoder.is_fitted:
            self.encoder_path.parent.mkdir(parents=True, exist_ok=True)
            self.sparse_encoder.save(self.encoder_path)
            print(f"[VectorStore] Sparse encoder saved")

        print(f"[VectorStore] Qdrant data persisted at {QDRANT_STORAGE_PATH}")

    def load_index(self, index_path: Path = None) -> bool:
        """Load sparse encoder and verify Qdrant collection exists

        Args:
            index_path: Not used (kept for API compatibility)

        Returns:
            True if loaded successfully, False otherwise
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        if not any(col.name == self.collection_name for col in collections):
            print(f"[VectorStore] Collection '{self.collection_name}' not found")
            return False

        self.collection_exists = True

        # Load sparse encoder if available
        if ENABLE_SPARSE_VECTORS and self.encoder_path.exists():
            success = self.sparse_encoder.load(self.encoder_path)
            if not success:
                print("[VectorStore] Warning: Could not load sparse encoder")

        # Get collection info
        collection_info = self.client.get_collection(self.collection_name)
        point_count = collection_info.points_count

        print(f"[VectorStore] Loaded Qdrant collection with {point_count} tickets")
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store

        Returns:
            Dictionary with collection statistics
        """
        if not self.collection_exists:
            return {'total_tickets': 0}

        try:
            collection_info = self.client.get_collection(self.collection_name)
            point_count = collection_info.points_count

            # Get category distribution by scrolling through points
            categories = {}
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust if more tickets
                with_payload=True
            )

            for point in scroll_result[0]:
                category = point.payload.get('category', 'Unknown')
                categories[category] = categories.get(category, 0) + 1

            return {
                'total_tickets': point_count,
                'embedding_dimension': QDRANT_VECTOR_SIZE,
                'categories': categories,
                'hybrid_search_enabled': ENABLE_SPARSE_VECTORS and self.sparse_encoder.is_fitted
            }

        except Exception as e:
            print(f"[VectorStore] Error getting statistics: {e}")
            return {'total_tickets': 0, 'error': str(e)}

"""Recommendation engine that orchestrates the entire ticket assistance pipeline"""
import time
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Document,
    SparseVectorParams, SparseIndexParams, Modifier,
    Prefetch, FusionQuery, Fusion
)
from src.data_loader import TicketDataLoader
from src.preprocessor import TextPreprocessor
from src.embedding_service import EmbeddingService
from src.llm_service import LLMService
from src.config import (
    QDRANT_STORAGE_PATH, QDRANT_COLLECTION_NAME, QDRANT_VECTOR_SIZE,
    TOP_K_RESULTS, ENABLE_SPARSE_VECTORS
)


class RecommendationEngine:
    """Orchestrates the ticket recommendation pipeline"""

    def __init__(
        self,
        embedding_service: EmbeddingService = None,
        llm_service: LLMService = None
    ):
        """
        Initialize recommendation engine

        Args:
            embedding_service: EmbeddingService instance (creates new if None)
            llm_service: LLMService instance (creates new if None)
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.llm_service = llm_service or LLMService()
        self.preprocessor = TextPreprocessor()

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(path=str(QDRANT_STORAGE_PATH))
        self.collection_name = QDRANT_COLLECTION_NAME
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Create Qdrant collection with dense + sparse vectors if it doesn't exist"""
        collections = self.qdrant_client.get_collections().collections
        if any(col.name == self.collection_name for col in collections):
            return

        # Dense vector configuration
        vectors_config = {
            "dense": VectorParams(size=QDRANT_VECTOR_SIZE, distance=Distance.COSINE)
        }

        # Sparse vector configuration for native BM25
        sparse_vectors_config = None
        if ENABLE_SPARSE_VECTORS:
            sparse_vectors_config = {
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(),
                    modifier=Modifier.IDF  # Native IDF calculation
                )
            }

        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config
        )

        # Create category payload index for filtering
        self.qdrant_client.create_payload_index(
            collection_name=self.collection_name,
            field_name="category",
            field_schema="keyword"
        )

        search_type = "hybrid (dense + sparse BM25)" if ENABLE_SPARSE_VECTORS else "dense-only"
        print(f"[RecommendationEngine] Created Qdrant collection '{self.collection_name}' with {search_type} search")

    def build_index(self, old_tickets: List[Dict[str, Any]]):
        """
        Build vector index from old tickets

        Args:
            old_tickets: List of resolved ticket dictionaries
        """
        print(f"\nBuilding index from {len(old_tickets)} tickets...")

        # Preprocess tickets
        print("Preprocessing tickets...")
        texts = [self.preprocessor.prepare_ticket_text(ticket) for ticket in old_tickets]

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_service.generate_embeddings_batch(
            texts,
            use_cache=True,
            show_progress=True
        )

        # Add to Qdrant with both dense and sparse vectors
        print("Adding to Qdrant...")
        points = []
        for idx, (ticket, embedding) in enumerate(zip(old_tickets, embeddings)):
            # Prepare text for BM25 encoding (issue + description)
            ticket_text = f"{ticket.get('issue', '')} {ticket.get('description', '')}"

            # Build vector configuration
            vector_config = {
                "dense": embedding.tolist()
            }

            # Add native BM25 sparse vector if enabled
            if ENABLE_SPARSE_VECTORS:
                vector_config["sparse"] = Document(text=ticket_text, model="Qdrant/bm25")

            points.append(PointStruct(
                id=idx,
                vector=vector_config,
                payload=ticket  # Store entire ticket as payload
            ))

        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)

        search_type = "hybrid (dense + sparse)" if ENABLE_SPARSE_VECTORS else "dense-only"
        print(f"[RecommendationEngine] Added {len(old_tickets)} tickets to Qdrant with {search_type} vectors")

        # Print statistics
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        print(f"\nIndex built successfully!")
        print(f"Total tickets: {collection_info.points_count}")
        print(f"Embedding dimension: {QDRANT_VECTOR_SIZE}")

        # Get category distribution
        category_stats = {}
        scroll_result = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True
        )
        for point in scroll_result[0]:
            category = point.payload.get('category', 'Unknown')
            category_stats[category] = category_stats.get(category, 0) + 1
        print(f"Categories: {category_stats}")

    def get_recommendation(self, new_ticket: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendation for a single new ticket

        Args:
            new_ticket: New ticket dictionary

        Returns:
            Dictionary with recommendation and metadata
        """
        start_time = time.time()

        # Preprocess ticket
        ticket_text = self.preprocessor.prepare_ticket_text(new_ticket)

        # Generate embedding
        query_embedding = self.embedding_service.generate_embedding(
            ticket_text,
            use_cache=True
        )

        # Prepare full text for BM25 (sparse vector query)
        full_text = f"{new_ticket.get('issue', '')} {new_ticket.get('description', '')}"

        # Execute search based on configuration
        if ENABLE_SPARSE_VECTORS:
            # Hybrid search with RRF fusion
            query_response = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    # Dense vector search (semantic)
                    Prefetch(
                        query=query_embedding.tolist(),
                        using="dense",
                        limit=TOP_K_RESULTS * 2  # Retrieve more candidates for fusion
                    ),
                    # Sparse vector search (BM25 keyword matching)
                    Prefetch(
                        query=Document(text=full_text, model="Qdrant/bm25"),
                        using="sparse",
                        limit=TOP_K_RESULTS * 2  # Retrieve more candidates for fusion
                    ),
                ],
                # RRF fusion combines both rankings
                query=FusionQuery(fusion=Fusion.RRF),
                limit=TOP_K_RESULTS,
                with_payload=True
            )
        else:
            # Dense-only search (fallback)
            query_response = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                using="dense",
                limit=TOP_K_RESULTS,
                with_payload=True
            )

        # Transform to expected format
        similar_tickets = [
            {'ticket': dict(point.payload), 'similarity_score': float(point.score)}
            for point in query_response.points
        ]

        # Generate recommendation using LLM
        if similar_tickets:
            recommendation = self.llm_service.generate_recommendation(
                new_ticket,
                similar_tickets
            )
        else:
            recommendation = "No similar tickets found in the database. Please escalate to a senior agent."

        # Calculate processing time
        processing_time = time.time() - start_time

        # Format result
        result = {
            'ticket_id': new_ticket.get('ticket_id'),
            'issue': new_ticket.get('issue'),
            'description': new_ticket.get('description'),
            'category': new_ticket.get('category'),
            'similar_tickets': [
                {
                    'ticket_id': st['ticket'].get('ticket_id'),
                    'issue': st['ticket'].get('issue'),
                    'description': st['ticket'].get('description'),
                    'category': st['ticket'].get('category'),
                    'resolution': st['ticket'].get('resolution'),
                    'agent_name': st['ticket'].get('agent_name'),
                    'resolved': st['ticket'].get('resolved', False),
                    'similarity_score': st['similarity_score']
                }
                for st in similar_tickets
            ],
            'recommendation': recommendation,
            'processing_time_seconds': round(processing_time, 2)
        }

        return result

    def process_all_new_tickets(
        self,
        new_tickets: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process all new tickets and generate recommendations

        Args:
            new_tickets: List of new ticket dictionaries
            show_progress: Whether to print progress

        Returns:
            List of recommendation results
        """
        results = []

        for i, ticket in enumerate(new_tickets):
            if show_progress:
                print(f"\nProcessing ticket {i + 1}/{len(new_tickets)}: {ticket.get('ticket_id')}...")

            try:
                result = self.get_recommendation(ticket)
                results.append(result)

                if show_progress:
                    print(f"[OK] Generated recommendation in {result['processing_time_seconds']}s")

            except Exception as e:
                print(f"[ERROR] Error processing ticket {ticket.get('ticket_id')}: {e}")
                # Add error result
                results.append({
                    'ticket_id': ticket.get('ticket_id'),
                    'issue': ticket.get('issue'),
                    'error': str(e)
                })

        return results

    def save_state(self):
        """Save embedding cache to disk (Qdrant auto-persists)"""
        print("\nSaving state...")
        self.embedding_service.save_cache()
        # Qdrant auto-persists to disk, no manual save needed
        print(f"State saved successfully (Qdrant data at {QDRANT_STORAGE_PATH})")

    def load_state(self) -> bool:
        """
        Load embedding cache and verify Qdrant collection exists

        Returns:
            True if state loaded successfully, False otherwise
        """
        print("\nLoading saved state...")
        self.embedding_service.load_cache()

        # Verify collection exists
        collections = self.qdrant_client.get_collections().collections
        if not any(col.name == self.collection_name for col in collections):
            print("No saved state found - collection doesn't exist")
            return False

        # Get collection info
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        point_count = collection_info.points_count

        print(f"State loaded successfully - {point_count} tickets indexed")
        return True

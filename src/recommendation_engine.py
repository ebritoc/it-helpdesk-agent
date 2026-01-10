"""Recommendation engine that orchestrates the entire ticket assistance pipeline"""
import time
from typing import List, Dict, Any
from src.data_loader import TicketDataLoader
from src.preprocessor import TextPreprocessor
from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.llm_service import LLMService


class RecommendationEngine:
    """Orchestrates the ticket recommendation pipeline"""

    def __init__(
        self,
        embedding_service: EmbeddingService = None,
        vector_store: VectorStore = None,
        llm_service: LLMService = None
    ):
        """
        Initialize recommendation engine

        Args:
            embedding_service: EmbeddingService instance (creates new if None)
            vector_store: VectorStore instance (creates new if None)
            llm_service: LLMService instance (creates new if None)
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()
        self.llm_service = llm_service or LLMService()
        self.preprocessor = TextPreprocessor()

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

        # Add to vector store
        print("Adding to vector store...")
        self.vector_store.add_tickets(old_tickets, embeddings)

        # Print statistics
        stats = self.vector_store.get_statistics()
        print(f"\nIndex built successfully!")
        print(f"Total tickets: {stats['total_tickets']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print(f"Categories: {stats['categories']}")

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

        # Search for similar tickets (with hybrid search)
        similar_tickets = self.vector_store.search(
            query_embedding,
            query_text=ticket_text  # Enable hybrid search
        )

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
        """Save embedding cache and vector index to disk"""
        print("\nSaving state...")
        self.embedding_service.save_cache()
        self.vector_store.save_index()
        print("State saved successfully")

    def load_state(self) -> bool:
        """
        Load embedding cache and vector index from disk

        Returns:
            True if state loaded successfully, False otherwise
        """
        print("\nLoading saved state...")
        self.embedding_service.load_cache()
        index_loaded = self.vector_store.load_index()

        if index_loaded:
            stats = self.vector_store.get_statistics()
            print(f"State loaded successfully - {stats['total_tickets']} tickets indexed")
            return True
        else:
            print("No saved state found")
            return False

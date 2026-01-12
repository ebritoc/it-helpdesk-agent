"""Test to see the actual prompt sent to the LLM"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.recommendation_engine import RecommendationEngine
from src.preprocessor import TextPreprocessor

print('Inspecting LLM prompt for TCKT-2004...')
print('=' * 70)

engine = RecommendationEngine()
engine.load_state()

# TCKT-2004 ticket
test_ticket = {
    'ticket_id': 'TCKT-2004',
    'issue': 'Cannot open files in shared drive',
    'description': 'User reports an error when trying to open files in the shared drive.',
    'category': 'Network'
}

# Prepare ticket text
preprocessor = TextPreprocessor()
ticket_text = preprocessor.prepare_ticket_text(test_ticket)

# Get embedding and search
query_embedding = engine.embedding_service.generate_embedding(ticket_text, use_cache=True)
similar_tickets = engine.vector_store.search(
    query_embedding,
    query_text=ticket_text
)

# Build prompt (same logic as in llm_service)
prompt = engine.llm_service._build_prompt(test_ticket, similar_tickets)

print('PROMPT SENT TO LLM:')
print('=' * 70)
print(prompt)
print('=' * 70)

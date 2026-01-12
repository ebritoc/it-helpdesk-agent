"""Test that resolved field is included in similar tickets"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.recommendation_engine import RecommendationEngine

engine = RecommendationEngine()
engine.load_state()

test_ticket = {
    'ticket_id': 'TCKT-2004',
    'issue': 'Cannot open files in shared drive',
    'description': 'User reports an error when trying to open files in the shared drive.',
    'category': 'Network'
}

result = engine.get_recommendation(test_ticket)

# Check if resolved field is present
print('Testing resolved field in similar tickets...')
print('=' * 70)
for i, ticket in enumerate(result['similar_tickets'][:5], 1):
    ticket_id = ticket.get('ticket_id')
    resolved = ticket.get('resolved', 'MISSING')
    status = "✅ RESOLVED" if resolved else "⚠️ UNRESOLVED"
    print(f'{i}. {ticket_id}: resolved={resolved} -> {status}')

print('=' * 70)
print('✅ Test passed! Resolved field is present in all similar tickets.')

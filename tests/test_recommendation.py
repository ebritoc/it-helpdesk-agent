"""Test recommendation generation with unresolved ticket warning"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.recommendation_engine import RecommendationEngine

print('Testing recommendation generation for TCKT-2004...')
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

print(f'\nQuery: {test_ticket["issue"]} (Category: {test_ticket["category"]})\n')

# Generate recommendation
recommendation = engine.get_recommendation(test_ticket)

print('=' * 70)
print('RECOMMENDATION:')
print('=' * 70)
print(recommendation)
print('=' * 70)

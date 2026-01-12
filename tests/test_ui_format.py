"""Test UI formatting with resolved status"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.recommendation_engine import RecommendationEngine
from app import format_result_as_markdown

engine = RecommendationEngine()
engine.load_state()

test_ticket = {
    'ticket_id': 'TCKT-2004',
    'issue': 'Cannot open files in shared drive',
    'description': 'User reports an error when trying to open files in the shared drive.',
    'category': 'Network'
}

result = engine.get_recommendation(test_ticket)

# Format the result as markdown
markdown_output = format_result_as_markdown(result)

print('UI Markdown Output Preview:')
print('=' * 70)
print(markdown_output[:1500])  # Show first 1500 chars
print('\n... (truncated)')
print('=' * 70)

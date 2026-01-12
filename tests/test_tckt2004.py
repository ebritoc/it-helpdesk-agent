"""Test script to verify TCKT-2004 matching improvements"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.recommendation_engine import RecommendationEngine

print('Testing TCKT-2004 matching with category boosting...')
print('=' * 70)

engine = RecommendationEngine()
print('\nLoading state...')
engine.load_state()

# TCKT-2004 ticket
test_ticket = {
    'ticket_id': 'TCKT-2004',
    'issue': 'Cannot open files in shared drive',
    'description': 'User reports an error when trying to open files in the shared drive.',
    'category': 'Network'
}

print(f'\n📋 Query Ticket: {test_ticket["ticket_id"]}')
print(f'   Issue: {test_ticket["issue"]}')
print(f'   Category: {test_ticket["category"]}')
print(f'\n🔍 Top 5 Similar Tickets:')
print('-' * 70)

result = engine.get_recommendation(test_ticket)

# Check if TCKT-1047 is in results
tckt1047_found = False
tckt1047_rank = 0

for i, st in enumerate(result['similar_tickets'], 1):
    category_match = '✅' if st['category'] == test_ticket['category'] else '  '

    if st['ticket_id'] == 'TCKT-1047':
        tckt1047_found = True
        tckt1047_rank = i
        marker = '🎯 TARGET MATCH!'
    else:
        marker = ''

    print(f'{i}. {st["ticket_id"]}: {st["issue"]} {marker}')
    print(f'   Category: {st["category"]} {category_match}')
    print(f'   Similarity: {st["similarity_score"]:.3f}')
    if i <= 3:
        print(f'   Resolution: {st.get("resolution", "N/A")[:60]}...')
    print()

print('=' * 70)
print(f'⏱️  Processing time: {result["processing_time_seconds"]}s')
print()

# Summary
if tckt1047_found:
    print(f'✅ SUCCESS: TCKT-1047 found at rank #{tckt1047_rank}!')
else:
    print(f'❌ ISSUE: TCKT-1047 (Shared drive access issue) NOT in top 5 results')
    print('   Expected: TCKT-1047 should match as it\'s about shared drive access')

print('\n📊 Category Distribution:')
same_category_count = sum(1 for st in result['similar_tickets'] if st['category'] == test_ticket['category'])
print(f'   Same category (Network): {same_category_count}/{len(result["similar_tickets"])}')
print(f'   Different category: {len(result["similar_tickets"]) - same_category_count}/{len(result["similar_tickets"])}')

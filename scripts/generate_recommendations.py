"""Script to generate recommendations for new tickets"""
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import TicketDataLoader
from src.recommendation_engine import RecommendationEngine
from src.config import RECOMMENDATIONS_JSON_PATH, RECOMMENDATIONS_TXT_PATH


def format_text_output(results):
    """Format results as human-readable text"""
    output = []
    output.append("=" * 80)
    output.append("IT HELPDESK TICKET RECOMMENDATIONS")
    output.append("=" * 80)

    for i, result in enumerate(results, 1):
        output.append(f"\n{'=' * 80}")
        output.append(f"TICKET {i}: {result.get('ticket_id')}")
        output.append("=" * 80)

        if 'error' in result:
            output.append(f"\n✗ ERROR: {result['error']}\n")
            continue

        output.append(f"\nIssue: {result.get('issue')}")
        output.append(f"Description: {result.get('description')}")
        output.append(f"Category: {result.get('category')}")

        output.append(f"\n{'-' * 80}")
        output.append("SIMILAR TICKETS:")
        output.append("-" * 80)

        for j, similar in enumerate(result.get('similar_tickets', []), 1):
            output.append(f"\n{j}. {similar.get('ticket_id')} (Similarity: {similar.get('similarity_score'):.3f})")
            output.append(f"   Issue: {similar.get('issue')}")
            output.append(f"   Category: {similar.get('category')}")
            output.append(f"   Resolution: {similar.get('resolution')}")
            output.append(f"   Resolved by: {similar.get('agent_name')}")

        output.append(f"\n{'-' * 80}")
        output.append("RECOMMENDATION:")
        output.append("-" * 80)
        output.append(result.get('recommendation', 'N/A'))

        output.append(f"\nProcessing time: {result.get('processing_time_seconds')}s")

    output.append("\n" + "=" * 80)
    output.append(f"Total tickets processed: {len(results)}")
    output.append("=" * 80)

    return "\n".join(output)


def main():
    """Generate recommendations for all new tickets"""
    print("=" * 80)
    print("IT Helpdesk Ticket Assistance System - Recommendation Generator")
    print("=" * 80)

    # Initialize components
    print("\nInitializing components...")
    data_loader = TicketDataLoader()
    engine = RecommendationEngine()

    # Try to load existing index
    index_loaded = engine.load_state()

    if not index_loaded:
        print("\n⚠ No existing index found. Building new index...")
        old_tickets = data_loader.load_all_old_tickets(filter_resolved=True)

        if not old_tickets:
            print("\n[ERROR] No resolved tickets found!")
            print("Please check that the data/old_tickets/ directory contains ticket files.")
            sys.exit(1)

        engine.build_index(old_tickets)
        engine.save_state()

    # Load new tickets
    print("\nLoading new tickets...")
    new_tickets = data_loader.load_new_tickets()

    if not new_tickets:
        print("\n[ERROR] No new tickets found!")
        print("Please check that data/new_tickets.csv exists.")
        sys.exit(1)

    # Process all new tickets
    print(f"\nProcessing {len(new_tickets)} new tickets...")
    print("=" * 80)

    results = engine.process_all_new_tickets(new_tickets, show_progress=True)

    # Save results as JSON
    print(f"\n\nSaving results to {RECOMMENDATIONS_JSON_PATH}...")
    with open(RECOMMENDATIONS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save results as text
    print(f"Saving results to {RECOMMENDATIONS_TXT_PATH}...")
    text_output = format_text_output(results)
    with open(RECOMMENDATIONS_TXT_PATH, 'w', encoding='utf-8') as f:
        f.write(text_output)

    # Print summary to console
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = sum(1 for r in results if 'error' not in r)
    failed = len(results) - successful

    print(f"\nTotal tickets processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if successful > 0:
        avg_time = sum(r.get('processing_time_seconds', 0) for r in results if 'error' not in r) / successful
        print(f"Average processing time: {avg_time:.2f}s")

    print("\n" + "=" * 80)
    print("[SUCCESS] Recommendations generated successfully!")
    print("=" * 80)
    print(f"\nOutputs saved to:")
    print(f"  - JSON: {RECOMMENDATIONS_JSON_PATH}")
    print(f"  - Text: {RECOMMENDATIONS_TXT_PATH}")

    # Print first recommendation as example
    if results and 'error' not in results[0]:
        print("\n" + "=" * 80)
        print("EXAMPLE RECOMMENDATION (First Ticket):")
        print("=" * 80)
        print(f"\nTicket: {results[0]['ticket_id']}")
        print(f"Issue: {results[0]['issue']}")
        print(f"\nRecommendation:")
        print(results[0]['recommendation'][:500] + "..." if len(results[0]['recommendation']) > 500 else results[0]['recommendation'])


if __name__ == "__main__":
    main()

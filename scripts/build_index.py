"""Script to build and save the vector index from old tickets"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import TicketDataLoader
from src.recommendation_engine import RecommendationEngine


def main():
    """Build vector index from resolved tickets"""
    print("=" * 60)
    print("IT Helpdesk Ticket Assistance System - Index Builder")
    print("=" * 60)

    # Initialize components
    print("\nInitializing components...")
    data_loader = TicketDataLoader()
    engine = RecommendationEngine()

    # Load old tickets (only resolved ones)
    print("\nLoading old tickets...")
    old_tickets = data_loader.load_all_old_tickets(filter_resolved=True)

    if not old_tickets:
        print("\n[ERROR] No resolved tickets found!")
        print("Please check that the data/old_tickets/ directory contains ticket files.")
        sys.exit(1)

    # Show category statistics
    print("\nCategory Distribution:")
    category_stats = data_loader.get_category_statistics(old_tickets)
    for category, count in sorted(category_stats.items()):
        print(f"  {category}: {count} tickets")

    # Build index
    engine.build_index(old_tickets)

    # Save to disk
    engine.save_state()

    print("\n" + "=" * 60)
    print("[SUCCESS] Index built and saved successfully!")
    print("=" * 60)
    print("\nNext step: Run 'python scripts/generate_recommendations.py' to generate recommendations")


if __name__ == "__main__":
    main()

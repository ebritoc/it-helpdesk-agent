"""Text preprocessing module for ticket data"""
from typing import Dict, Any


class TextPreprocessor:
    """Handles text preprocessing for ticket data before embedding"""

    @staticmethod
    def combine_fields(issue: str, description: str) -> str:
        """
        Combine issue and description into a single text for embedding

        Args:
            issue: Brief issue title
            description: Detailed problem description

        Returns:
            Combined text representation
        """
        # Create semantic text by combining issue and description
        combined = f"Issue: {issue}. Description: {description}"
        return combined

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Basic text cleaning

        Args:
            text: Input text to clean

        Returns:
            Cleaned text
        """
        # Strip leading/trailing whitespace
        text = text.strip()

        # Normalize multiple spaces to single space
        text = ' '.join(text.split())

        return text

    def prepare_ticket_text(self, ticket: Dict[str, Any]) -> str:
        """
        Prepare ticket text for embedding generation

        Args:
            ticket: Ticket dictionary with 'issue' and 'description' fields

        Returns:
            Preprocessed text ready for embedding
        """
        issue = ticket.get('issue', '')
        description = ticket.get('description', '')

        # Combine fields
        combined_text = self.combine_fields(issue, description)

        # Clean text
        cleaned_text = self.clean_text(combined_text)

        return cleaned_text

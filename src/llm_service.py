"""LLM service for generating resolution recommendations using HuggingFace API"""
import time
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
from src.config import (
    HF_API_TOKEN,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_NEW_TOKENS,
    MAX_RETRIES,
    RETRY_DELAY,
    BACKOFF_FACTOR
)


class LLMService:
    """Handles LLM-based recommendation generation via HuggingFace Inference API"""

    def __init__(self, api_token: str = None, model_name: str = None):
        """
        Initialize LLM service

        Args:
            api_token: HuggingFace API token (defaults to config)
            model_name: Model name (defaults to config)
        """
        self.api_token = api_token or HF_API_TOKEN
        self.model_name = model_name or LLM_MODEL
        self.client = InferenceClient(token=self.api_token)

    def _build_prompt(
        self,
        new_ticket: Dict[str, Any],
        similar_tickets: List[Dict[str, Any]]
    ) -> str:
        """
        Build prompt for LLM with new ticket and similar resolved tickets

        Args:
            new_ticket: New ticket dictionary
            similar_tickets: List of similar ticket dictionaries with similarity scores

        Returns:
            Formatted prompt string
        """
        prompt = "You are an IT helpdesk expert assistant. Your task is to recommend a resolution for a new ticket based on similar previously resolved tickets.\n\n"

        # Add new ticket information
        prompt += "NEW TICKET:\n"
        prompt += f"Ticket ID: {new_ticket.get('ticket_id', 'N/A')}\n"
        prompt += f"Issue: {new_ticket.get('issue', 'N/A')}\n"
        prompt += f"Description: {new_ticket.get('description', 'N/A')}\n"
        prompt += f"Category: {new_ticket.get('category', 'N/A')}\n\n"

        # Add similar tickets
        prompt += "SIMILAR RESOLVED TICKETS:\n\n"

        for i, similar in enumerate(similar_tickets, 1):
            ticket = similar.get('ticket', {})
            score = similar.get('similarity_score', 0.0)

            prompt += f"Similar Ticket {i} (Similarity: {score:.2f}):\n"
            prompt += f"Ticket ID: {ticket.get('ticket_id', 'N/A')}\n"
            prompt += f"Issue: {ticket.get('issue', 'N/A')}\n"
            prompt += f"Description: {ticket.get('description', 'N/A')}\n"
            prompt += f"Category: {ticket.get('category', 'N/A')}\n"
            prompt += f"Resolution: {ticket.get('resolution', 'N/A')}\n"
            prompt += f"Resolved by: {ticket.get('agent_name', 'N/A')}\n\n"

        # Add instruction
        prompt += "Based on these similar resolved tickets, provide a specific and actionable resolution recommendation for the new ticket. "
        prompt += "Focus on practical steps the helpdesk agent can take to resolve the issue.\n\n"
        prompt += "RECOMMENDATION:"

        return prompt

    def _call_api(self, prompt: str) -> str:
        """
        Make API call to generate text with retry logic

        Args:
            prompt: Prompt text for generation

        Returns:
            Generated text

        Raises:
            Exception: If API call fails after retries
        """
        for attempt in range(MAX_RETRIES):
            try:
                # Use InferenceClient's chat_completion method for instruction-tuned models
                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

                response = self.client.chat_completion(
                    messages,
                    model=self.model_name,
                    max_tokens=LLM_MAX_NEW_TOKENS,
                    temperature=LLM_TEMPERATURE
                )

                # Extract the response content
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    result = response.choices[0].message.content
                    return result.strip()
                else:
                    raise ValueError(f"Unexpected API response format: {type(response)}")

            except Exception as e:
                error_msg = str(e)
                print(f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {error_msg}")

                # Check if it's a model loading error
                if "503" in error_msg or "loading" in error_msg.lower():
                    wait_time = RETRY_DELAY * (attempt + 2)
                    print(f"Model loading, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue

                # Other errors: retry with exponential backoff
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (BACKOFF_FACTOR ** attempt)
                    time.sleep(wait_time)
                    continue

                # Last attempt failed
                raise

        raise Exception(f"Failed to generate recommendation after {MAX_RETRIES} attempts")

    def generate_recommendation(
        self,
        new_ticket: Dict[str, Any],
        similar_tickets: List[Dict[str, Any]]
    ) -> str:
        """
        Generate resolution recommendation for a new ticket

        Args:
            new_ticket: New ticket dictionary
            similar_tickets: List of similar ticket dictionaries with similarity scores

        Returns:
            Generated recommendation text
        """
        # Build prompt
        prompt = self._build_prompt(new_ticket, similar_tickets)

        # Generate recommendation
        recommendation = self._call_api(prompt)

        return recommendation

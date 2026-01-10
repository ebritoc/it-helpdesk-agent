"""Gradio web application for IT Helpdesk Ticket Recommendation System"""
import gradio as gr
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.recommendation_engine import RecommendationEngine
from src.data_loader import TicketDataLoader

# Global variables
engine = None
evaluation_tickets = []
evaluation_tickets_dict = {}


def initialize():
    """Initialize engine and load data at startup"""
    global engine, evaluation_tickets, evaluation_tickets_dict

    print("Initializing IT Helpdesk Recommendation System...")

    # Initialize recommendation engine
    engine = RecommendationEngine()

    # Load saved vector index
    if not engine.load_state():
        error_msg = """
        ❌ **Error: No saved vector index found!**

        Please run `python scripts/build_index.py` first to build the index from resolved tickets.
        """
        print("[ERROR] No saved index found. Run build_index.py first.")
        return False, error_msg

    print("[OK] Vector index loaded successfully")

    # Load evaluation tickets
    try:
        loader = TicketDataLoader()
        evaluation_tickets = loader.load_new_tickets()

        # Create dict for quick lookup
        for ticket in evaluation_tickets:
            ticket_id = ticket.get('ticket_id', 'UNKNOWN')
            evaluation_tickets_dict[ticket_id] = ticket

        print(f"[OK] Loaded {len(evaluation_tickets)} evaluation tickets")
    except Exception as e:
        print(f"[WARNING] Could not load evaluation tickets: {e}")
        evaluation_tickets = []

    return True, None


def format_result_as_markdown(result):
    """Format recommendation result as markdown with expandable similar tickets"""

    # Extract data
    recommendation_text = result.get('recommendation', 'No recommendation available')
    processing_time = result.get('processing_time_seconds', 0)
    similar_tickets = result.get('similar_tickets', [])

    # Build markdown output
    output = f"""## 🎯 AI-Powered Recommendation

{recommendation_text}

---

**⏱️ Processing Time:** {processing_time:.2f} seconds | **📊 Similar Tickets Found:** {len(similar_tickets)}

---

## 📋 Similar Resolved Tickets

"""

    # Add similar tickets as expandable HTML details
    for i, ticket_info in enumerate(similar_tickets):
        # ticket_info already contains the ticket fields at top level (not nested)
        ticket = ticket_info
        similarity_score = ticket_info.get('similarity_score', 0) * 100

        # First ticket is open by default
        open_attr = ' open' if i == 0 else ''

        output += f"""
<details{open_attr}>
<summary><b>🎫 Similar Ticket #{i+1} - {ticket.get('ticket_id', 'N/A')} (Similarity: {similarity_score:.1f}%)</b></summary>

**Issue:** {ticket.get('issue', 'N/A')}

**Category:** {ticket.get('category', 'N/A')}

**Description:** {ticket.get('description', 'N/A')}

**Resolution:** {ticket.get('resolution', 'N/A')}

**Resolved By:** {ticket.get('agent_name', 'N/A')}

</details>
"""

    if not similar_tickets:
        output += "\n*No similar resolved tickets found in the database.*\n"

    return output


def process_manual_ticket(issue, description, category):
    """Process manually entered ticket (Tab 1)"""

    # Validate inputs
    if not issue or not issue.strip():
        return "⚠️ **Error:** Please provide an issue title."

    if not description or not description.strip():
        return "⚠️ **Error:** Please provide a detailed description."

    # Category is optional - "All Categories" means search all categories
    # No validation needed for category

    try:
        # Create ticket dict
        ticket_id = f"MANUAL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # If "All Categories" is selected, set category to None to search all
        actual_category = None if category == "All Categories" else category

        ticket = {
            'ticket_id': ticket_id,
            'issue': issue.strip(),
            'description': description.strip(),
            'category': actual_category
        }

        # Show processing message
        gr.Info("Processing your ticket... This may take 5-8 seconds.")

        # Get recommendation
        result = engine.get_recommendation(ticket)

        # Format output
        output_markdown = format_result_as_markdown(result)

        return output_markdown

    except Exception as e:
        error_output = f"""## ❌ Error

An error occurred while processing your ticket:

```
{str(e)}
```

Please try again or check the application logs for more details.
"""
        return error_output


def get_ticket_dropdown_choices():
    """Get dropdown choices for evaluation tickets"""
    if not evaluation_tickets:
        return []

    choices = []
    for ticket in evaluation_tickets:
        ticket_id = ticket.get('ticket_id', 'UNKNOWN')
        issue = ticket.get('issue', 'No issue')
        choices.append(f"{ticket_id}: {issue}")

    return choices


def on_ticket_select(ticket_choice):
    """Update fields when ticket selected (Tab 2)"""
    if not ticket_choice:
        return "", "", "", ""

    # Parse ticket ID from choice (format: "TCKT-2000: Issue title")
    ticket_id = ticket_choice.split(':')[0].strip()

    # Find ticket
    ticket = evaluation_tickets_dict.get(ticket_id)

    if not ticket:
        return "Ticket not found", "", "", ""

    return (
        ticket.get('issue', ''),
        ticket.get('description', ''),
        ticket.get('category', ''),
        ticket.get('date', '')
    )


def process_evaluation_ticket(ticket_choice):
    """Process selected evaluation ticket (Tab 2)"""
    if not ticket_choice:
        return "⚠️ **Error:** Please select a ticket from the dropdown."

    # Parse ticket ID
    ticket_id = ticket_choice.split(':')[0].strip()

    # Find ticket
    ticket = evaluation_tickets_dict.get(ticket_id)

    if not ticket:
        return "⚠️ **Error:** Selected ticket not found."

    try:
        # Show processing message
        gr.Info(f"Processing ticket {ticket_id}... This may take 5-8 seconds.")

        # Get recommendation
        result = engine.get_recommendation(ticket)

        # Format output
        output_markdown = format_result_as_markdown(result)

        return output_markdown

    except Exception as e:
        error_output = f"""## ❌ Error

An error occurred while processing ticket {ticket_id}:

```
{str(e)}
```

Please try again or check the application logs for more details.
"""
        return error_output


def clear_inputs():
    """Clear all input fields in Tab 1"""
    return "", "", "All Categories"


# Initialize the system
success, error_msg = initialize()

# Build Gradio interface
with gr.Blocks(
    title="IT Helpdesk AI Assistant",
    theme=gr.themes.Soft(),
    css="""
    .output-markdown { min-height: 400px; }
    .tab-label { font-weight: bold; }
    """
) as demo:

    gr.Markdown("""
    # 🎫 IT Helpdesk Ticket Recommendation System

    Get AI-powered recommendations for IT support tickets based on similar resolved cases.
    """)

    if not success:
        gr.Markdown(error_msg)
    else:
        with gr.Tabs():
            # Tab 1: Try It Yourself
            with gr.Tab("✨ Try It Yourself"):
                gr.Markdown("""
                ### Enter Your IT Issue
                Describe your technical problem and get instant recommendations based on similar resolved tickets.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        issue_input = gr.Textbox(
                            label="Issue Title",
                            placeholder="Brief description of the problem (e.g., 'VPN connection timeout')",
                            lines=1
                        )

                        description_input = gr.Textbox(
                            label="Detailed Description",
                            placeholder="Provide detailed information about the issue, including any error messages, when it started, etc.",
                            lines=5
                        )

                        category_input = gr.Dropdown(
                            label="Category",
                            choices=["All Categories", "Network", "Software", "Hardware", "Account Management"],
                            value="All Categories"
                        )

                        with gr.Row():
                            submit_btn = gr.Button("Get Recommendation", variant="primary", size="lg")
                            clear_btn = gr.Button("Clear", variant="secondary")

                    with gr.Column(scale=1):
                        output_manual = gr.Markdown(
                            label="Recommendation",
                            value="*Enter your ticket details and click 'Get Recommendation' to see results.*",
                            elem_classes=["output-markdown"]
                        )

                # Event handlers for Tab 1
                submit_btn.click(
                    fn=process_manual_ticket,
                    inputs=[issue_input, description_input, category_input],
                    outputs=output_manual
                )

                clear_btn.click(
                    fn=clear_inputs,
                    outputs=[issue_input, description_input, category_input]
                )

            # Tab 2: Evaluation Dataset
            with gr.Tab("📊 Evaluation Dataset"):
                gr.Markdown("""
                ### Test with Real Tickets
                Select a ticket from our evaluation dataset to see how the system performs on real cases.
                """)

                if not evaluation_tickets:
                    gr.Markdown("""
                    ⚠️ **No evaluation tickets found.**

                    Please ensure `data/new_tickets.csv` exists and contains tickets.
                    """)
                else:
                    ticket_dropdown = gr.Dropdown(
                        label="Select a Ticket",
                        choices=get_ticket_dropdown_choices(),
                        value=None
                    )

                    gr.Markdown("### Ticket Details")

                    with gr.Row():
                        with gr.Column():
                            eval_issue = gr.Textbox(label="Issue", interactive=False)
                            eval_description = gr.Textbox(label="Description", lines=3, interactive=False)

                        with gr.Column():
                            eval_category = gr.Textbox(label="Category", interactive=False)
                            eval_date = gr.Textbox(label="Date", interactive=False)

                    process_eval_btn = gr.Button(
                        "Get Recommendation for This Ticket",
                        variant="primary",
                        size="lg"
                    )

                    gr.Markdown("### Results")

                    output_eval = gr.Markdown(
                        value="*Select a ticket and click 'Get Recommendation' to see results.*",
                        elem_classes=["output-markdown"]
                    )

                    # Event handlers for Tab 2
                    ticket_dropdown.change(
                        fn=on_ticket_select,
                        inputs=ticket_dropdown,
                        outputs=[eval_issue, eval_description, eval_category, eval_date]
                    )

                    process_eval_btn.click(
                        fn=process_evaluation_ticket,
                        inputs=ticket_dropdown,
                        outputs=output_eval
                    )

        gr.Markdown("""
        ---

        **About this system:**
        This AI-powered recommendation system uses RAG (Retrieval-Augmented Generation) to find similar resolved tickets
        and generate actionable recommendations. It combines semantic search with large language models to provide
        context-aware support suggestions.

        **Models used:**
        - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
        - LLM: `meta-llama/Llama-3.1-8B-Instruct`
        """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public Gradio link
        show_error=True
    )

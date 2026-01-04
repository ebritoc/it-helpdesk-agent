# IT Helpdesk Ticket Assistance System

An AI-powered system that assists IT helpdesk agents by finding similar resolved tickets and generating resolution recommendations using semantic search and large language models.

## Overview

This system uses a **Retrieval-Augmented Generation (RAG)** approach to help IT helpdesk agents resolve tickets faster by:

1. **Semantic Search**: Finding previously resolved tickets that are similar to new issues
2. **AI Recommendations**: Generating specific, actionable resolution guidance based on past solutions
3. **Multi-format Support**: Loading ticket data from CSV, Excel, and JSON files

## Architecture

```
New Ticket Input
    ↓
Text Preprocessing (Issue + Description)
    ↓
Embedding Generation (sentence-transformers/all-MiniLM-L6-v2)
    ↓
Cosine Similarity Search
    ↓
Retrieve Top-3 Similar Resolved Tickets
    ↓
LLM Prompt Construction
    ↓
Resolution Recommendation (meta-llama/Llama-3.1-8B-Instruct)
    ↓
Multi-format Output (JSON + Text + Console)
```

### Key Components

- **Data Loader**: Unified loader for CSV, XLSX, and JSON ticket formats
- **Text Preprocessor**: Combines ticket fields for optimal semantic representation
- **Embedding Service**: Generates 384-dimensional vectors via HuggingFace API
- **Vector Store**: In-memory cosine similarity search with numpy
- **LLM Service**: Generates resolution recommendations via HuggingFace API
- **Recommendation Engine**: Orchestrates the entire pipeline

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- HuggingFace account (free)
- Internet connection for API access

### 2. Get HuggingFace API Token

1. Go to [HuggingFace](https://huggingface.co/) and create a free account
2. Navigate to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Click "New token"
4. Give it a name (e.g., "IT-Helpdesk-System")
5. Select "Read" permission
6. Click "Generate token"
7. Copy the token (you'll need it in step 4)

### 3. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your HuggingFace API token
# Replace 'your_huggingface_api_token_here' with your actual token
```

Your `.env` file should look like:
```
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Usage

### Step 1: Build the Index

This step processes all resolved tickets and creates a searchable vector index:

```bash
python scripts/build_index.py
```

**What it does:**
- Loads all old tickets from `data/old_tickets/` (CSV, XLSX, JSON)
- Filters to only include tickets marked as Resolved=True
- Generates embeddings for each ticket
- Saves the index to `outputs/vector_index.pkl` and `outputs/embeddings_cache.pkl`

**Output:**
```
Loaded 10 tickets from ticket_dump_1.csv
Loaded 11 tickets from ticket_dump_2.xlsx
Loaded 10 tickets from ticket_dump_3.json
Filtered to 21 resolved tickets

Building index from 21 tickets...
Generating embeddings...
✓ Index built successfully!
```

### Step 2: Generate Recommendations

Process new tickets and generate recommendations:

```bash
python scripts/generate_recommendations.py
```

**What it does:**
- Loads the pre-built index (or builds it if not found)
- Processes each new ticket from `data/new_tickets.csv`
- Finds the 3 most similar resolved tickets
- Generates AI-powered resolution recommendations
- Saves results in multiple formats

**Outputs:**
- `outputs/recommendations.json` - Structured JSON with all details
- `outputs/recommendations.txt` - Human-readable text format
- Console output with summary and example

## Project Structure

```
it-helpdesk-agent/
├── data/
│   ├── new_tickets.csv              # New tickets to process
│   └── old_tickets/                 # Resolved tickets database
│       ├── ticket_dump_1.csv
│       ├── ticket_dump_2.xlsx
│       └── ticket_dump_3.json
├── src/
│   ├── __init__.py
│   ├── config.py                    # Configuration and constants
│   ├── data_loader.py               # Multi-format data loading
│   ├── preprocessor.py              # Text preprocessing
│   ├── embedding_service.py         # HuggingFace embedding API
│   ├── vector_store.py              # Similarity search
│   ├── llm_service.py               # LLM recommendation API
│   └── recommendation_engine.py     # Pipeline orchestration
├── scripts/
│   ├── build_index.py               # Index builder script
│   └── generate_recommendations.py  # Recommendation generator
├── outputs/                         # Generated files
│   ├── embeddings_cache.pkl         # Cached embeddings
│   ├── vector_index.pkl             # Vector search index
│   ├── recommendations.json         # JSON output
│   └── recommendations.txt          # Text output
├── .env                             # Environment variables (create this)
├── .env.example                     # Template for .env
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md                        # This file
```

## How It Works

### 1. Semantic Search with Embeddings

The system uses `sentence-transformers/all-MiniLM-L6-v2` to convert ticket text into 384-dimensional vectors. These embeddings capture the semantic meaning of the text, allowing the system to find similar tickets even when they use different words.

**Example:**
- "VPN connection timeout" and "VPN keeps disconnecting" would have high similarity
- Category alone isn't enough - "Email not syncing" and "Email client crashes" are both Software but need different solutions

### 2. Retrieval-Augmented Generation (RAG)

Rather than asking the LLM to guess solutions, we:
1. Find 3 actually resolved tickets that are similar
2. Show the LLM what worked before
3. Ask it to generate a specific recommendation based on those examples

This grounds the LLM's output in real resolutions, making recommendations more accurate and actionable.

### 3. Prompt Engineering

The system constructs prompts that include:
- New ticket details (ID, Issue, Description, Category)
- Similar tickets with their resolutions
- Similarity scores to indicate confidence
- Clear instructions for actionable recommendations

## Example Output

```json
{
  "ticket_id": "TCKT-2000",
  "issue": "VPN connection timeout",
  "description": "VPN connection times out frequently during use.",
  "category": "Network",
  "similar_tickets": [
    {
      "ticket_id": "TCKT-1011",
      "issue": "VPN disconnection issues",
      "similarity_score": 0.873,
      "resolution": "VPN settings updated",
      "category": "Network"
    }
  ],
  "recommendation": "Based on the similar VPN disconnection case, I recommend the following steps:\n\n1. Update VPN client settings...",
  "processing_time_seconds": 2.3
}
```

## Evaluation

### Quality Assessment

To evaluate the system:

1. **Relevance of Similar Tickets**: Check if retrieved tickets are actually similar
2. **Category Alignment**: Do similar tickets match the category?
3. **Resolution Quality**: Are recommendations specific and actionable?
4. **Consistency**: Do similar new tickets get similar recommendations?

### Metrics

- **Similarity Scores**: Higher scores (>0.7) indicate strong matches
- **Processing Time**: Typically 2-5 seconds per ticket
- **Coverage**: % of new tickets with high-similarity matches

## Future Improvements

### Short-term (5-10 hours)
- **Hybrid Search**: Combine semantic similarity with category filtering
- **Prompt Engineering**: Add few-shot examples and chain-of-thought reasoning
- **Confidence Scores**: Indicate when the system is uncertain
- **Web UI**: Build a Streamlit or Gradio interface for easier use

### Long-term (Production System)
- **Vector Database**: Use Pinecone, Weaviate, or ChromaDB for scalability
- **Fine-tuned Models**: Train embedding model on IT helpdesk domain data
- **User Feedback Loop**: Let agents rate recommendations to improve over time
- **Multi-stage Retrieval**: Use cheap model first, then rerank with better model
- **Auto-categorization**: Automatically categorize incoming tickets
- **REST API**: FastAPI endpoint for integration with existing helpdesk software
- **Monitoring**: Track usage, performance, and recommendation quality

## Troubleshooting

### API Token Issues

**Error:** `HF_API_TOKEN not found in environment variables`

**Solution:** Make sure you've created a `.env` file with your HuggingFace token:
```bash
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Model Loading Errors

**Error:** `503 - Model is loading`

**Solution:** The HuggingFace API needs to load the model. The system will automatically wait and retry. First requests may take 30-60 seconds.

### Rate Limiting

**Error:** `429 - Rate limit exceeded`

**Solution:** The free tier has rate limits. The system includes retry logic with exponential backoff. For production use, consider:
- HuggingFace Pro subscription
- Self-hosted models
- Caching (already implemented)

### No Similar Tickets Found

**Issue:** System returns "No similar tickets found"

**Possible causes:**
1. No tickets in the index (check `build_index.py` output)
2. Similarity threshold too high (adjust in `config.py`)
3. New ticket category has no examples in old tickets

## Technical Details

### Models Used

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
  - 384-dimensional vectors
  - Fast and efficient for semantic search
  - Good balance of quality and speed

- **Generation**: `meta-llama/Llama-3.1-8B-Instruct`
  - 8B parameter instruction-tuned model
  - Good at following instructions
  - Generates coherent, contextual recommendations

### Configuration

Key parameters (in `src/config.py`):

```python
TOP_K_RESULTS = 3              # Number of similar tickets to retrieve
SIMILARITY_THRESHOLD = 0.0     # Minimum similarity score (0-1)
LLM_TEMPERATURE = 0.7          # Higher = more creative, lower = more focused
LLM_MAX_NEW_TOKENS = 500       # Maximum length of recommendations
```

### Data Schema

**New Tickets:**
- Ticket ID, Issue, Description, Category, Date

**Old Tickets:**
- Ticket ID, Issue, Description, Category, Date, Resolution, Agent Name, Resolved

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

This is a prototype system built for evaluation purposes. For production use, consider the improvements listed in the "Future Improvements" section.

## Support

For questions or issues:
1. Check the Troubleshooting section
2. Review HuggingFace API documentation
3. Check that all dependencies are installed correctly

---

**Built with:**
- HuggingFace Inference API
- sentence-transformers
- Llama 3.1
- Python, NumPy, Pandas

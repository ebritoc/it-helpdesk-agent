# IT Helpdesk Ticket Assistance System

An AI-powered system that helps IT helpdesk agents resolve tickets faster by finding similar past cases and generating actionable recommendations using hybrid search and large language models.

## What It Does

- 🔍 **Hybrid Search**: Finds similar tickets using both semantic similarity (dense vectors) and keyword matching (BM25 sparse vectors)
- 🤖 **AI Recommendations**: Generates specific resolution guidance using Llama 3.1 based on past solutions
- ⚠️ **Smart Warnings**: Alerts when recommendations come from unresolved tickets
- 🌐 **Web Interface**: Interactive Gradio UI for easy ticket processing
- 💾 **Persistent Storage**: Uses Qdrant vector database for production-grade performance

## Quick Start

### 1. Prerequisites

- Python 3.8+
- HuggingFace account (free) - [Sign up here](https://huggingface.co/)

### 2. Get HuggingFace API Token

1. Go to [HuggingFace Settings → Access Tokens](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permission
3. Copy the token for the next step

### 3. Install

```bash
# Clone or download the repository
cd it-helpdesk-agent

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your token: HF_API_TOKEN=hf_xxxxxxxxxxxxx
```

### 4. Build Index

```bash
python scripts/build_index.py
```

This processes all tickets (resolved and unresolved) from `data/old_tickets/` and builds a searchable index with:
- Dense embeddings (384-dim vectors from sentence-transformers/all-MiniLM-L6-v2)
- Sparse BM25 vectors for keyword matching
- Saves to `outputs/qdrant_storage/`

### 5. Use the System

**Option A: Web Interface (Recommended)**

```bash
python app.py
```

Open http://localhost:7860 in your browser. Two tabs available:
- **Try It Yourself**: Enter ticket details manually
- **Evaluation Dataset**: Test with pre-loaded tickets

**Option B: Command Line**

```bash
python scripts/generate_recommendations.py
```

Processes tickets from `data/new_tickets.csv` and saves results to:
- `outputs/recommendations.json` (structured data)
- `outputs/recommendations.txt` (human-readable)

## How It Works

```
New Ticket Input
    ↓
Text Preprocessing (Issue + Description)
    ↓
Embedding Generation (sentence-transformers/all-MiniLM-L6-v2)
    ↓
Hybrid Search (Qdrant):
  ├─ Dense Vector Search (semantic similarity)
  └─ Sparse Vector Search (BM25 keyword matching)
         ↓
    Reciprocal Rank Fusion (RRF) combines rankings
    ↓
Retrieve Top-5 Similar Tickets (resolved + unresolved)
    ↓
LLM Prompt with Warnings for Unresolved Tickets
    ↓
Resolution Recommendation (meta-llama/Llama-3.1-8B-Instruct)
```

**Key Features:**

- **Hybrid Search**: Combines semantic understanding with keyword matching for better retrieval
- **RRF Fusion**: Reciprocal Rank Fusion algorithm balances dense and sparse search results
- **Unresolved Tickets**: System includes unresolved tickets but warns when using them as reference
- **Persistent Storage**: Qdrant vector database for production-ready performance
- **Status Badges**: UI shows resolved/unresolved status for each similar ticket

## Configuration

Key settings in `src/config.py`:

```python
TOP_K_RESULTS = 5              # Number of similar tickets to retrieve
SIMILARITY_THRESHOLD = 0.0     # Minimum similarity score (0-1)
ENABLE_SPARSE_VECTORS = True   # Enable BM25 keyword search
BM25_K1 = 1.2                  # BM25 term saturation
BM25_B = 0.75                  # BM25 length normalization
```

## Project Structure

```
it-helpdesk-agent/
├── data/
│   ├── new_tickets.csv              # New tickets to process
│   └── old_tickets/                 # Historical tickets (resolved + unresolved)
├── src/
│   ├── config.py                    # Configuration
│   ├── vector_store.py              # Qdrant + hybrid search
│   ├── sparse_encoder.py            # BM25 implementation
│   ├── recommendation_engine.py     # Pipeline orchestration
│   └── ...
├── scripts/
│   ├── build_index.py               # Build search index
│   └── generate_recommendations.py  # Batch processing
├── app.py                           # Gradio web interface
├── outputs/
│   └── qdrant_storage/              # Vector database
└── .env                             # API token (create this)
```

## Data Format

The system loads tickets from the `data/` directory. Supported formats: **CSV**, **Excel (.xlsx)**, and **JSON**.

### Historical Tickets (for index building)

Place files in `data/old_tickets/`:

**Required fields:**
- `Ticket ID` - Unique identifier (e.g., "TCKT-1001")
- `Issue` - Brief issue title (e.g., "VPN connection timeout")
- `Description` - Detailed description
- `Category` - Category name (e.g., "Network", "Software", "Hardware")
- `Date` - Date string (e.g., "2024-01-15")
- `Resolution` - How it was resolved
- `Agent Name` - Who resolved it
- `Resolved` - Boolean: `True`, `False`, `1`, `0`, `"true"`, `"false"`

**Example CSV (data/old_tickets/tickets.csv):**
```csv
Ticket ID,Issue,Description,Category,Date,Resolution,Agent Name,Resolved
TCKT-1001,VPN timeout,VPN disconnects after 5 minutes,Network,2024-01-15,Updated VPN settings,John Doe,True
TCKT-1002,Printer offline,Cannot print documents,Hardware,2024-01-16,Reinstalled drivers,Jane Smith,True
TCKT-1003,Email sync issue,Emails not syncing,Software,2024-01-17,Investigating root cause,Bob Jones,False
```

**Example JSON (data/old_tickets/tickets.json):**
```json
[
  {
    "Ticket ID": "TCKT-1001",
    "Issue": "VPN timeout",
    "Description": "VPN disconnects after 5 minutes",
    "Category": "Network",
    "Date": "2024-01-15",
    "Resolution": "Updated VPN settings",
    "Agent Name": "John Doe",
    "Resolved": true
  }
]
```

### New Tickets (to process)

Place in `data/new_tickets.csv`:

**Required fields:**
- `Ticket ID`, `Issue`, `Description`, `Category`, `Date`

**Example:**
```csv
Ticket ID,Issue,Description,Category,Date
TCKT-2001,Cannot access shared drive,User reports permission denied error,Network,2024-02-01
```

**Note:** Field names are case-insensitive. Both `Ticket ID` and `ticket_id` work.

## Testing from Scratch

To test everything from a clean state:

```bash
# 1. Clean previous outputs
rm -rf outputs/

# 2. Rebuild index (loads 30 tickets: 17 resolved, 13 unresolved)
python scripts/build_index.py

# 3. Test with evaluation dataset
python scripts/generate_recommendations.py

# 4. Launch web interface
python app.py
# Open http://localhost:7860 and select "TCKT-2004" from Evaluation Dataset
```

**Expected Results:**
- Index builds with 30 tickets
- TCKT-2004 matches with similar shared drive tickets
- TCKT-1047 appears in results (marked as UNRESOLVED)
- Web UI shows status badges: ✅ RESOLVED or ⚠️ UNRESOLVED

## Troubleshooting

**API Token Error**
```
Error: HF_API_TOKEN not found
Solution: Create .env file with HF_API_TOKEN=your_token_here
```

**Model Loading (503 Error)**
```
Error: Model is loading
Solution: Wait 30-60 seconds, system auto-retries
```

**Database Lock Error**
```
Error: Storage folder already accessed
Solution: Close other Python processes using the database
```

## Technical Details

**Models:**
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- Generation: `meta-llama/Llama-3.1-8B-Instruct`

**Search Algorithm:**
- Hybrid search with dense (semantic) + sparse (BM25 keyword) vectors
- Reciprocal Rank Fusion (RRF) for combining results
- Category-based metadata filtering support

**Dependencies:**
- `qdrant-client>=1.7.0` - Vector database
- `gradio>=4.0.0` - Web interface
- `pandas`, `numpy`, `requests` - Data processing
- `python-dotenv` - Environment configuration

## License

MIT License - see LICENSE file for details

---

**Built with:** HuggingFace API • Qdrant • Llama 3.1 • sentence-transformers • Gradio

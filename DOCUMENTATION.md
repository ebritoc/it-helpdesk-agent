# IT Helpdesk Ticket Assistance System
## Case Study Documentation

**Author:** Eduardo Brito Chacón
**Date:** January 13 2025  

---

## Executive Summary

This document presents a prototype for an IT helpdesk ticket assistance system that helps service agents resolve incoming tickets by leveraging historical case data. The system uses **Retrieval-Augmented Generation (RAG)** with **hybrid search** (semantic + keyword-based) to find similar past tickets and generate actionable recommendations.

---

## 1. Problem Understanding

### 1.1 Business Context

The client operates IT helpdesks for multiple companies. Key challenges identified:

1. **Redundant tickets**: Many incoming issues are similar or identical to previously resolved cases
2. **Knowledge silos**: Agents don't know how colleagues solved similar problems before
3. **Inefficient resolution**: Time wasted reinventing solutions that already exist
4. **Inconsistent quality**: Resolution approaches vary by agent experience

### 1.2 Solution Objectives

The goal is **not** to automate ticket resolution, but to **assist agents** by:

- Finding relevant past tickets that match the new issue
- Providing resolution guidance based on what worked before
- Highlighting when similar tickets remain unresolved (to warn agents)

### 1.3 Key Assumptions

Based on the provided data and problem statement:

1. **Historical tickets contain valuable resolution information** that can guide future cases
2. **Semantic similarity captures issue relatedness** better than exact keyword matching alone
3. **Both resolved and unresolved tickets are informative** — unresolved tickets show what doesn't work or are still work-in-progress
4. **Agents need direction, not automation** — the system augments human judgment
5. **Response time matters** — agents need recommendations within seconds, not minutes

---

## 2. Solution Architecture

### 2.1 High-Level Design

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           IT Helpdesk Assistant                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────────┐       ┌────────────────────────┐  │
│  │   New Ticket │────▶│Text Preprocessor│──────▶│  Embedding Generation  │  │
│  │    Input     │     │   (Issue + Desc) │       │ (all-MiniLM-L6-v2)     │  │
│  └──────────────┘     └──────────────────┘       └─────────┬──────────────┘  │
│                                                            │                 │
│                              ┌─────────────────────────────▼──────────────┐  │
│                              │      HYBRID SEARCH (Qdrant)                │  │
│                              │  ┌──────────────────────────────────────┐  │  │
│                              │  │ Dense: Semantic (Cosine, 384-dim)    │  │  │
│                              │  │ Sparse: BM25 Keyword (native)        │  │  │
│                              │  └──────────────────┬───────────────────┘  │  │
│                              │  ┌──────────────────▼───────────────────┐  │  │
│                              │  │ RRF Fusion (reciprocal rank combo)   │  │  │
│                              │  └──────────────────┬───────────────────┘  │  │
│                              └────────────────────┼───────────────────────┘  │
│                                                   │                          │
│                              ┌────────────────────▼───────────────────────┐  │
│                              │       Top-K Similar Tickets                │  │
│                              │   (Resolved + Unresolved with warnings)    │  │
│                              └──────────────────┬─────────────────────────┘  │
│                                                 │                            │
│                              ┌──────────────────▼────────────────────────┐   │
│                              │  LLM Recommendation (Llama 3.1 8B)        │   │
│                              │  - Analyzes similar tickets               │   │
│                              │  - Generates actionable guidance          │   │
│                              │  - Warns about unresolved references      │   │
│                              └──────────────────┬────────────────────────┘   │
│                                                 │                            │
│                              ┌──────────────────▼────────────────────────┐   │
│                              │       Agent-Friendly Output               │   │
│                              │  - Recommendation text                    │   │
│                              │  - Similar tickets with details           │   │
│                              │  - Status badges (Resolved/Unresolved)    │   │
│                              └───────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Vector Database** | Qdrant (>=1.15.2) | Built-in hybrid search with native BM25 and RRF fusion, metadata filtering, production-grade, direct client integration |
| **Dense Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Lightweight, fast, good semantic quality, 384 dimensions |
| **Sparse Vectors** | Qdrant native BM25 (fastembed) | Keyword matching for exact terms (product names, error codes), no custom encoder needed |
| **Search Fusion** | Qdrant RRF | Reciprocal Rank Fusion automatically balances dense + sparse results |
| **Search Method** | Hybrid (Dense + Sparse with RRF) | Best of both worlds: semantic understanding + keyword precision |
| **LLM** | meta-llama/Llama-3.1-8B-Instruct | Open-source, instruction-tuned, good reasoning |
| **Web Interface** | Gradio | Rapid prototyping, built-in features |
| **Data Processing** | Pandas | Flexible input format support (CSV, XLSX, JSON) |

### 2.3 Retrieval Approach

**Hybrid Search with Native BM25**

The system uses Qdrant's native capabilities for hybrid search:
- **Dense vectors**: Captures semantics. Strong capturing synonyms, abstracts concepts (384-dim)
- **Sparse vectors**: Native BM25. Strong for keyword matching (product names, error codes)
- **RRF Fusion**: Built-in reciprocal rank fusion balances both approaches
- **Direct Qdrant Client**: No abstraction layers, simplified codebase
- **No custom encoder**: Qdrant + fastembed handle BM25 automatically

**Benefits of Hybrid Search Native Implementation:**
- ✅ Best of both worlds: semantic + keyword matching
- ✅ Zero custom encoder code needed
- ✅ Direct access to all Qdrant features
- ✅ Automatic RRF fusion (no manual tuning)
- ✅ Simple architecture with minimal dependencies

---

## 3. Implementation Details

### 3.1 Data Pipeline

```python
# Data loading supports multiple formats
data/old_tickets/
├── tickets.csv          # CSV format
├── archive.xlsx         # Excel format
└── legacy.json          # JSON format

# Normalized schema
{
    'ticket_id': 'TCKT-1001',
    'issue': 'VPN connection timeout',
    'description': 'VPN disconnects after 5 minutes...',
    'category': 'Network',
    'date': '2024-01-15',
    'resolution': 'Updated VPN settings to...',
    'agent_name': 'John Doe',
    'resolved': True  # Boolean: True/False
}
```

### 3.2 Text Preprocessing

Combines `Issue` and `Description` fields for embedding:

```python
def prepare_ticket_text(ticket):
    issue = ticket.get('issue', '')
    description = ticket.get('description', '')
    return f"{issue}. {description}"
```

This approach:
- Preserves context from both fields
- Allows semantic matching on either component
- Keeps preprocessing minimal to avoid information loss

### 3.3 Embedding Service

Uses HuggingFace Inference API with:
- **Batch processing**: Efficient API calls for index building
- **Caching**: Avoids redundant API calls
- **Retry logic**: Handles transient failures gracefully

```python
# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
```

### 3.4 Native Hybrid Search with RRF Fusion

**Hybrid Search Architecture:**

The system uses Qdrant's built-in capabilities for hybrid search:

**Dense Vectors (Semantic)**:
- 384-dimensional embeddings from all-MiniLM-L6-v2
- Captures meaning, handles synonyms, abstracts concepts
- Cosine similarity for scoring

**Sparse Vectors (Keyword)**:
- Qdrant's native BM25 implementation via fastembed
- `Document(text=..., model="Qdrant/bm25")` for automatic encoding
- Server-side IDF calculation with `Modifier.IDF`
- Handles exact matches: product names, error codes, technical terms

**RRF Fusion**:
- Prefetch runs dense + sparse queries in parallel
- `FusionQuery(fusion=Fusion.RRF)` combines rankings
- Reciprocal rank scoring: score = Σ(1 / (60 + rank))
- No manual weight tuning needed

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Document,
    SparseVectorParams, SparseIndexParams, Modifier,
    Prefetch, FusionQuery, Fusion
)

# Initialize client
client = QdrantClient(path="./qdrant_storage")

# Create collection with hybrid vectors
client.create_collection(
    collection_name="helpdesk_tickets",
    vectors_config={
        "dense": VectorParams(size=384, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(
            index=SparseIndexParams(),
            modifier=Modifier.IDF
        )
    }
)

# Index tickets with both dense and sparse vectors
ticket_text = f"{ticket['issue']} {ticket['description']}"
points = [
    PointStruct(
        id=idx,
        vector={
            "dense": embedding.tolist(),
            "sparse": Document(text=ticket_text, model="Qdrant/bm25")
        },
        payload=ticket
    )
    for idx, (ticket, embedding) in enumerate(zip(tickets, embeddings))
]
client.upsert(collection_name="helpdesk_tickets", points=points)

# Hybrid search with RRF fusion
full_text = f"{query_ticket['issue']} {query_ticket['description']}"
results = client.query_points(
    collection_name="helpdesk_tickets",
    prefetch=[
        Prefetch(query=query_embedding.tolist(), using="dense", limit=6),
        Prefetch(query=Document(text=full_text, model="Qdrant/bm25"), using="sparse", limit=6),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=3
)
```

**Benefits:**
- **Best of both worlds**: Semantic understanding + keyword precision
- **Zero custom encoder code**: Qdrant + fastembed handle BM25 automatically
- **Built-in RRF fusion**: No manual score combination needed
- **No abstraction overhead**: Direct API calls to Qdrant
- **Full feature access**: All Qdrant capabilities immediately available

### 3.5 RecommendationEngine (Core Logic)

Key features:
- **Persistent storage**: Data survives restarts via Qdrant
- **Direct client integration**: Uses QdrantClient without abstraction layers
- **Hybrid search**: Combines dense semantic + sparse BM25 with RRF fusion
- **Metadata filtering**: Category-based pre-filtering via payload index
- **Payload storage**: Full ticket data stored with vectors

```python
# Search configuration
TOP_K_RESULTS = 3
ENABLE_SPARSE_VECTORS = True  # Enable hybrid search with native BM25
# Qdrant's native BM25 with RRF fusion (no manual tuning needed)
```

### 3.6 LLM Recommendation Generation

Structured prompt engineering:

```
You are an IT helpdesk expert assistant...

NEW TICKET:
Ticket ID: TCKT-2004
Issue: Cannot access shared drive
Description: User reports permission denied error...

SIMILAR TICKETS (RESOLVED AND UNRESOLVED):

Similar Ticket 1 (Similarity: 0.85) - STATUS: RESOLVED:
...
Resolution: Reset network credentials and...

Similar Ticket 2 (Similarity: 0.72) - STATUS: UNRESOLVED:
...
⚠️ WARNING: This ticket was NOT fully resolved...

Based on these similar tickets, provide a specific and actionable 
resolution recommendation...
```

---

## 4. Experimentation & Results

### 4.1 Dataset Analysis

| Metric | Value |
|--------|-------|
| Total old tickets | 30 |
| Resolved tickets | 17 |
| Unresolved tickets | 13 |
| New tickets (evaluation) | 10 |
| Categories | Network, Software, Hardware, Account Management |

### 4.2 Search Quality Observations

Testing with evaluation tickets revealed:

**Strengths:**
- Semantic matches across phrasings: "VPN timeout" ↔ "VPN disconnects"
- Keyword matches for exact terms: "Outlook", "Active Directory", error codes
- Hybrid search balances both approaches via RRF fusion
- Category-relevant tickets consistently ranked high

**Weaknesses:**
- Small corpus limits retrieval diversity
- Some edge cases with ambiguous descriptions
- Cross-category matches occasionally noisy

### 4.3 Sample Results

**Test Case: TCKT-2004 - "Cannot access shared drive"**

| Rank | Ticket | Similarity | Status |
|------|--------|------------|--------|
| 1 | TCKT-1023 | 87% | ✅ Resolved |
| 2 | TCKT-1047 | 75% | ⚠️ Unresolved |
| 3 | TCKT-1012 | 68% | ✅ Resolved |

**Generated Recommendation:**
> "Based on similar resolved cases, start by verifying the user's Active Directory group membership and checking the share permissions. If the user recently changed roles, their access may need to be updated. Note: Similar ticket TCKT-1047 remains unresolved — the attempted permission reset didn't fully solve the issue, suggesting a deeper AD synchronization problem."

---

## 5. Shortcomings & Limitations

### 5.1 Technical Limitations

1. **Small training corpus**: 30 tickets is insufficient for robust semantic coverage
2. **No fine-tuning**: Using off-the-shelf embedding model without domain adaptation
3. **Single-language support**: Only English tickets tested
4. **No feedback loop**: System doesn't learn from agent corrections

### 5.2 Architecture Limitations

1. **Synchronous processing**: No async/batch recommendation generation
2. **No caching layer**: LLM calls made for every request
3. **Single-node deployment**: No horizontal scaling considerations
4. **No authentication**: Open access to the interface
5. **Tailored for Qdrant**: Migrating to a different vector DB may need a architecture adaptation

### 5.3 Evaluation Limitations

1. **No ground truth labels**: Cannot compute precision/recall metrics
2. **Subjective quality**: "Good" recommendations are tester-dependent
3. **Limited test set**: Only 10 evaluation tickets

---

## 6. Future Improvements

### 6.1 Short-term 

1. **Evaluation framework**: Create labeled test set with expected similar tickets
2. **Async LLM calls**: Non-blocking recommendation generation
3. **Result caching**: Cache recommendations for identical queries
4. **Better error handling**: Graceful degradation when services unavailable

### 6.2 Medium-term

1. **Domain-specific embeddings**: Fine-tune embedding model on IT helpdesk corpus
2. **Re-ranking model**: Add cross-encoder for result refinement
3. **Feedback integration**: Agent can mark recommendations as helpful/unhelpful
4. **Multi-language support**: Embed and recommend in multiple languages if we may expect non-English tickets.
5. **Assess value of late-interaction bag-of-embeddings**: Evaluate ColBERT-style late interaction models for better semantic matching with token-level precision.

### 6.3 Long-term (3+ months)

1. **Continuous learning**: Update embeddings as new tickets are resolved
2. **Knowledge graph**: Build structured relationships between issues
3. **Proactive recommendations**: Suggest resolutions before agents ask
4. **Resolution automation**: Auto-apply known fixes for high-confidence cases

---

## 7. Design Decisions & Trade-offs

### 7.1 Including Unresolved Tickets

**Decision:** Index both resolved AND unresolved tickets

**Rationale:**
- Given the small corpus size, many new tickets have no semantically similar resolved tickets -> Finding similar unresolved tickets is more informative than unrelated resolved tickets (and demonstrates that retrieval works)
- Unresolved tickets may show what doesn't work or potential resulution still in progress
- System explicitly warns when referencing unresolved cases

**Trade-off:** May surface incomplete information, but transparency mitigates risk


### 7.2 Free-tier API vs Local Models

**Decision:** Use HuggingFace free API (as instructed)

**Rationale:**
- Zero infrastructure cost for prototyping
- Easy to swap for local models in production
- Sufficient for demonstration purposes

**Trade-off:** Rate limits and latency, but acceptable for this use case

---

## 8. Production Readiness Checklist

| Aspect | Status | Notes |
|--------|--------|-------|
| Persistent storage | ✅ Done | Qdrant with local persistence |
| Error handling | ✅ Done | Retry logic for API calls |
| Logging | ⚠️ Basic | Console output only |
| Monitoring | ❌ Missing | No metrics collection |
| Authentication | ❌ Missing | Open access |
| Rate limiting | ❌ Missing | No request throttling |
| API documentation | ⚠️ Partial | README covers usage |
| Unit/integration tests | ❌ Missing | No automated tests |
| CI/CD | ❌ Missing | Manual deployment only |

---

## 9. How to Run

### Prerequisites
- Python 3.8+
- HuggingFace account with API token
- Qdrant server v1.15.2 or newer (for native BM25 support)

### Installation
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your HF_API_TOKEN
```

### Build Index
```bash
python scripts/build_index.py
```

### Architecture Note

**Hybrid Search with Native BM25:**

The system uses Qdrant's native capabilities for hybrid search. Benefits include:
- Best of both worlds: semantic + keyword matching
- Native BM25 implementation (no custom encoder needed)
- Built-in RRF fusion (automatic ranking combination)
- Direct Qdrant client integration
- Simple architecture with minimal dependencies

**Requirements:**
1. `qdrant-client>=1.15.2` - Vector database with hybrid search support
2. `fastembed>=0.2.0` - Required for native BM25 sparse vectors

**Migration from previous versions:**
1. Upgrade dependencies: `pip install -U qdrant-client>=1.15.2 fastembed>=0.2.0`
2. Delete old Qdrant collection: `rm -rf outputs/qdrant_storage/`
3. Rebuild index: `python scripts/build_index.py`
4. Old files no longer needed: `src/vector_store.py`, `src/sparse_encoder.py`

### Run Web Interface
```bash
python app.py
# Open http://localhost:7860
```

### Run Batch Processing
```bash
python scripts/generate_recommendations.py
# Results in outputs/recommendations.json
```

---

## 10. Conclusion

This solution demonstrates a pragmatic approach to the IT helpdesk assistance problem:

1. **Hybrid search** with dense + sparse vectors balances semantic understanding and keyword precision
2. **Native BM25 implementation** via Qdrant eliminates custom encoder complexity from other solutions
3. **Built-in RRF fusion** automatically combines rankings without custom implementation
4. **RAG architecture** combines retrieval with LLM generation
5. **Transparent warnings** about unresolved references build agent trust
6. **Simplified architecture** with direct client integration reduces complexity

The focus was on delivering a **working, demonstrable system** rather than a perfect but incomplete solution. Future iterations would benefit from larger training data, domain-specific fine-tuning, and a proper evaluation framework.

---


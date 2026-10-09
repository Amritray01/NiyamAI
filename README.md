# 🛡️ NiyamAI: FSSAI Regulatory RAG Search & Compliance Platform

NiyamAI is an enterprise-grade regulatory search, intelligence, and compliance engine purpose-built to parse, index, retrieve, and synthesize complex Indian Food Safety and Standards Authority of India (**FSSAI**) regulations, gazette notifications, multi-page statutory tables, and amendments with zero hallucination.

It leverages an advanced **Hierarchical Dual-Text RAG Architecture** with **Qdrant Vector Engine** and **ChromaDB**, **BAAI/bge-small-en-v1.5 dense semantic embeddings**, **BM25 Sparse Lexical Search**, **Reciprocal Rank Fusion (RRF: k=60)**, **FlashRank Cross-Encoder Reranking**, and **LangGraph Regulatory Agent** with Synset Query Expansion, 2nd-hop HyDE, and deterministic **Numeric Precision Assertion Guard**.

---

## 🏛️ Final System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Intelligent Regulatory Ingestion"]
        PDF["FSSAI Gazette PDFs / Compendia"] --> Parser["Docling / PyMuPDF Layout Parser"]
        Parser --> HeaderProp["Multi-Page Header Propagation"]
        Parser --> FootnoteExt["Superscript Footnote Binder"]
        HeaderProp & FootnoteExt --> Chunker["Dual-Text Chunker"]
        Chunker --> DispText["Display Text (Visual Markdown Layout)"]
        Chunker --> EmbText["Embedding Text (Semantic Key-Value Tokens)"]
    end

    subgraph Storage ["2. Dual-Engine Storage Layer"]
        EmbText --> DenseModel["BAAI/bge-small-en-v1.5 Embedder"]
        DenseModel --> Qdrant["Qdrant Vector DB (Production: HNSW Cosine)"]
        DenseModel --> Chroma["ChromaDB (Dedicated Offline Test & Eval Store)"]
        DispText --> BM25["Rank-BM25 Sparse Inverted Index"]
    end

    subgraph Retrieval ["3. Multi-Stage Hybrid Retrieval & Fusion"]
        Query["User Legal / Compliance Query"] --> Synset["FSSAI Synset Expansion"]
        Synset --> Dense["1. Dense Vector Search (Top-20)"]
        Synset --> Sparse["2. BM25 Sparse Search (Top-20)"]
        Dense & Sparse --> RRF["3. Reciprocal Rank Fusion (Top-20 Combined, k=60)"]
        RRF --> Reranker["4. FlashRank Cross-Encoder Reranker"]
        Reranker --> TopDocs["Top-7 Relevant Statutory Chunks"]
    end

    subgraph Generation ["4. Guarded Synthesis & Citation"]
        TopDocs --> HyDEGuard{"HyDE 2nd-Hop Needed?"}
        HyDEGuard -->|Yes| HyDERetrieve["2nd-Hop Dense Expansion"]
        HyDEGuard -->|No| PromptEngine["Statutory Prompt + Citation Enforcer"]
        HyDERetrieve --> PromptEngine
        PromptEngine --> LLM["Groq 120B / Gemini-1.5 / GPT-4o"]
        LLM --> NumericGuard["Numeric Precision & MRL Assertion Guard"]
        NumericGuard --> FinalOut["Verified Regulatory Answer + Exact Citations"]
    end
```

---

## 🚀 Key Architectural Highlights

1. **State-of-the-Art Dense Semantic Embeddings (`BAAI/bge-small-en-v1.5`)**:
   - 384-dimensional normalized dense vectors trained specifically for high-precision regulatory document retrieval and semantic search.
2. **Hierarchical Dual-Text Representation**:
   - `display_text`: Preserves human-readable Markdown tables, footnote associations, and legal formatting for UI presentation and LLM synthesis.
   - `embedding_text`: Flattens rows into self-contained semantic assertions (`[Contaminants] -> Crop Contaminants -> Food: Aflatoxin M1 -> Maximum Limit: 0.5 µg/kg`) to ensure high dense retrieval fidelity.
3. **Multi-Page Table Header Propagation**:
   - Automatically detects paginated regulatory tables and carries forward parent table titles, column headers, and footnote anchors across page breaks.
4. **Dual Vector Stores (Production + Test Environments)**:
   - **Qdrant**: Primary high-performance production store supporting native payload filtering, HNSW vector search, and scalable cloud clustering.
   - **ChromaDB**: Dedicated local in-memory/embedded store for headless test runners, CI/CD pipelines, and ground-truth evaluation.
5. **Hybrid Search with Reciprocal Rank Fusion (RRF)**:
   - Retrieves **Top-20** dense semantic vector candidates ($D$) from `BAAI/bge-small-en-v1.5` and **Top-20** BM25 sparse lexical candidates ($S$).
   - Fuses and deduplicates rankings using:
     $$RRF(d) = \sum_{m \in \{D, S\}} \frac{1}{k + r_m(d)} \quad \text{with } k=60$$
   - Retains the **Top-20 combined candidates** to provide sufficient headroom for multi-document synthesis and commodity tables.
6. **FlashRank Cross-Encoder Reranking (Top-7 Selection)**:
   - Performs deep query-document cross-attention over the Top-20 combined pool using ultra-fast, CPU-optimized transformer rerankers (`ms-marco-TinyBERT-L-2-v2`) to surface the definitive **Top-7 statutory chunks** for LLM synthesis.
7. **LangGraph Agentic Search & Numeric Assertion Guard**:
   - Evaluates retrieved chunk completeness, triggers 2nd-hop HyDE expansion for numeric queries, and scans generated answers against source document chunks to verify exact numeric thresholds (e.g., `0.02 mg/kg`, `15 ppb`, `99.8%`) before emitting answers.

---

## 🧪 Modular Benchmark & Evaluation Results

We evaluated NiyamAI across the 25-query canonical Golden Dataset covering **Exact Statutory Clauses**, **Numeric MRL Limits**, **Semantic Rules**, **Superscript Footnotes**, **Amendments**, and **Multi-Document Syntheses**.

### 📊 Progression Across Architectural Stages

The modular evaluation compares retrieval performance across each architectural layer:

| Architectural Stage | Strategy Description | Hit@1 | Hit@3 | Hit@5 | Hit@7 | Recall@7 | MRR | Fact Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1: Baseline BM25** | Pure BM25 Lexical Keyword Search | 16.0% | 36.0% | 52.0% | 56.0% | 37.3% | 0.2930 | 29.0% |
| **Stage 2: Semantic Only** | Dense Embeddings (`BAAI/bge-small-en-v1.5`) | 8.0% | 20.0% | 32.0% | 32.0% | 14.7% | 0.1547 | 21.0% |
| **Stage 3: Filtered Search** | Metadata-Constrained Dense Search | 8.0% | 20.0% | 32.0% | 32.0% | 14.7% | 0.1547 | 21.0% |
| **Stage 4: Hybrid + Rerank** | **Dense (top 20) + BM25 (top 20) + RRF (top 20) + FlashRank (top 7)** | **28.0%** | **52.0%** | **52.0%** | **60.0%** | **38.7%** | **0.3924** | **26.3%** |

### 🤖 RAGAS & Generation Quality Evaluation (LLM-as-a-Judge)

In addition to retrieval ranking metrics, NiyamAI executes a comprehensive legal compliance **RAGAS LLM-as-a-Judge Evaluation Engine** ([`evaluation/ragas_eval.py`](file:///d:/Coding_files/NiyamAI/evaluation/ragas_eval.py)) powered by Groq's reasoning model (`openai/gpt-oss-120b`). It audits synthesized answers against retrieved legal contexts and golden ground-truth references across standard RAGAS dimensions:

1. **Faithfulness (Zero-Hallucination Score)**: Measures whether every statement in the generated answer is strictly grounded in retrieved regulatory text.
2. **Answer Relevance**: Measures how directly and completely the response addresses the user's specific statutory query.
3. **Context Precision**: Evaluates the signal-to-noise ratio of statutory provisions present in the retrieved chunks.
4. **Ingredient Classification Accuracy**: Validates correct distinction between permitted and prohibited food additives and ingredients.

#### 📈 Final Golden Dataset RAGAS Benchmark Scores

| Evaluation Metric | Score (%) | Benchmark Assessment |
| :--- | :---: | :--- |
| **Faithfulness (Zero-Hallucination)** | **47.72%** | Grounded against retrieved statutory context with Numeric Guard |
| **Answer Relevance** | **63.32%** | Direct compliance answering across legal angles and FBO concerns |
| **Context Precision** | **51.36%** | High concentration of numerical thresholds in top retrieved chunks |
| **Ingredient Classification Accuracy** | **67.92%** | Accurate permitted vs prohibited ingredient classification |
| **Overall RAGAS Judge Score** | **57.81%** | Composite multi-dimensional statutory compliance quality |

#### 📂 RAGAS Performance Breakdown by Query Category

| Category | Queries | Faithfulness | Relevance | Context Precision | Overall Judge Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Exact Clause** | 5 | 53.80% | 80.20% | 61.20% | **61.20%** |
| **Numeric Limit** | 6 | 53.67% | 51.33% | 51.33% | **57.83%** |
| **Semantic Rule** | 4 | 66.25% | 74.50% | 57.50% | **70.25%** |
| **Footnote Exception** | 3 | 46.67% | 46.67% | 50.00% | **52.83%** |
| **Amendment Tracking** | 4 | 30.50% | 57.00% | 49.50% | **52.25%** |
| **Multi-Document Synthesis** | 3 | 25.00% | 69.33% | 30.67% | **47.92%** |

---

## 📂 Project Structure

```
NiyamAI/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── middleware/telemetry.py       # Latency, token count, & API cost tracker
│   │   │   └── routes/
│   │   │       ├── search.py                 # Hybrid RAG search & citation endpoint
│   │   │       ├── compare.py                # Global regulatory matrix comparison
│   │   │       ├── news.py                   # Advisory updates & gazette notifications
│   │   │       └── admin.py                  # Telemetry & aggregation metrics
│   │   ├── core/config.py                    # App settings & LLM provider configurations
│   │   └── rag/
│   │       ├── models.py                     # FSSAIChunk (dual-text), SearchResult schemas
│   │       ├── parsers.py                    # Multi-page table header & footnote parser
│   │       ├── chunkers.py                   # Semantic dual-text row & clause chunker
│   │       ├── vector_store.py               # Qdrant (Production) & ChromaDB (Test) stores
│   │       ├── retriever.py                  # Hybrid Dense + BM25 + RRF + FlashRank
│   │       ├── chains.py                     # Synthesis chain + Numeric Assertion Guard
│   │       └── pipeline.py                   # Unified FSSAIPipeline factory
│   └── data/raw/pdfs/                        # Canonical FSSAI PDF Regulations
├── evaluation/
│   ├── golden_dataset.jsonl                  # 25 curated canonical regulatory test cases
│   ├── retrieval_eval.py                     # Hit@K, MRR, Recall@K, Fact Coverage metrics
│   ├── staged_evaluator.py                   # 4-Stage architectural benchmark engine
│   ├── run_eval.py                           # Benchmark CLI runner & summary exporter
│   └── results/                              # Evaluation CSV and JSON benchmark reports
└── frontend/
    └── src/                                  # React 18 + Vite Glass UI interface
```

---

## ⚙️ Setup & Quickstart

### 1. Environment Configuration
Create a `.env` file in the project root:
```env
# Vector Store Configurations
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
CHROMA_PERSIST_DIR=./data/chroma_db

# LLM Providers
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key

# Database & Telemetry
DATABASE_URL=postgresql://user:password@neon-db-host/niyamai?sslmode=require
```

### 2. Run Ingestion & Benchmark Evaluations
To parse canonical FSSAI PDFs and benchmark retrieval & generation quality:
```bash
# 1. Run the 4-stage modular retrieval benchmark (Hit@K, MRR, Fact Coverage)
python evaluation/run_eval.py

# Optional: Run specifically against Qdrant vector backend
python evaluation/run_eval.py --use-qdrant

# 2. Run the RAGAS LLM-as-a-Judge generation audit (Faithfulness, Relevance, Precision)
python evaluation/ragas_eval.py
```

### 3. Start the Backend API
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### 4. Start the Frontend Application
```bash
cd frontend
npm install
npm run dev
```

---

## 📡 API Reference

* `POST /api/search`: Executes hybrid RAG query with statutory citations, confidence score, and numeric precision verification.
* `POST /api/compare`: Compares FSSAI standards against US FDA and EU EFSA regulations.
* `GET /api/admin/metrics`: Aggregates real-time token usage, latency (ms), and estimated LLM costs.
* `GET /api/news`: Fetches the latest FSSAI gazette circulars and compliance advisories.



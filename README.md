# 🛡️ NiyamAI: FSSAI Regulatory Search & Compliance Platform

NiyamAI is an intelligent regulatory platform designed to transform food safety regulations, standard acts, and packaging guidelines into actionable, explainable compliance insights. 

It is engineered for food business operators (FBOs), compliance officers, and developers, leveraging a modern stack of **React, FastAPI, LangChain, Qdrant Cloud, NeonDB (PostgreSQL), and Ragas**.

---

## 🚀 Key Features

* **🔍 FSSAI Regulation Search & Citations**: Direct semantic and exact-lexical search across legal clauses with verified section numbers, page numbers, and verbatim text quotes.
* **🌐 Global Benchmark Matrix**: Side-by-side comparison of food standards across different regions (India/FSSAI vs. Europe/EFSA vs. USA/FDA).
* **📊 Admin Telemetry & Cost Engine**: Real-time request monitoring detailing execution time (ms), prompt/completion token usage, and calculated LLM API costs in USD.
* **🧠 User-Scoped sliding-window Memory**: Strict database-level isolation of conversation histories per user with sliding-window truncation to minimize token bloat and optimize latency.
* **📰 Regulatory News & Update Feed**: Centralized updates, FSSAI advisory circulars, and enforcement case notifications.
* **🧪 Ingestion & Evaluation Ready**: Starter blueprints for automatic PDF section chunking and Ragas-based faithfulness evaluation.

---

## 🛠️ Technology Stack

* **Frontend**: React (Vite, CSS Modules / Modern Glass UI)
* **Backend**: FastAPI (Python 3.10+)
* **Vector Database**: Qdrant Cloud (dense semantic indexing + hybrid search capability)
* **Relational Database**: NeonDB PostgreSQL (telemetry logs, user management, conversation memory)
* **Orchestration**: LangChain (structured LLM output parsing, memory framing, and prompts)
* **Evaluation**: Ragas (verification metrics for faithfulness and answer relevance)

---

## 📂 Project Structure

```
NiyamAI/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── middleware/
│   │   │   │   └── telemetry.py       # Tracks latency, tokens, & API costs
│   │   │   ├── routes/
│   │   │   │   ├── admin.py           # Admin metrics aggregator endpoint
│   │   │   │   ├── compare.py        # Global food safety comparison route
│   │   │   │   ├── news.py           # Advisory updates feed route
│   │   │   │   └── search.py          # Search query & citation logic
│   │   ├── core/
│   │   │   ├── config.py              # Configuration & settings definitions
│   │   │   ├── auth.py                # Simple User & Admin role access checks
│   │   │   └── telemetry.py           # In-memory/Postgres logging helpers
│   │   ├── rag/
│   │   │   ├── chains.py              # RAG orchestration & prompt mapping
│   │   │   └── vector_store.py        # Qdrant Cloud client & search helpers
│   │   ├── schemas/
│   │   │   └── regulatory.py          # Pydantic validation schemas
│   │   └── main.py                    # Main FastAPI app configuration
│   └── scripts/
│       └── ingest_fssai.py            # PDF document ingestion & chunking script
└── frontend/
    ├── src/
    │   ├── App.jsx                    # Primary React application hub
    │   ├── styles.css                 # Premium dark-mode glass styling
    │   └── main.jsx                   # React bootloader
```

---

## ⚙️ Setup & Configuration

### 1. Backend Setup
1. Move to the backend folder:
   ```bash
   cd backend
   ```
2. Copy the environment template:
   ```bash
   cp ../.env.example .env
   ```
3. Update `.env` with your cloud credentials:
   - **NeonDB Connection String**: Paste your connection string into `DATABASE_URL` (ensure `sslmode=require` is present).
   - **Qdrant Cloud Cluster**: Set your `QDRANT_URL` and `QDRANT_API_KEY`.
   - **Google or OpenAI API Key**: Add the appropriate LLM provider token.
4. Run the FastAPI development server:
   ```bash
   uvicorn app.main:app --reload
   ```

### 2. Frontend Setup
1. Move to the frontend folder:
   ```bash
   cd ../frontend
   ```
2. Install dependencies and start the Vite server:
   ```bash
   npm install
   npm run dev
   ```

---

## 📡 API Reference & Development Roadmap

1. **POST `/api/search`**: Process queries, query Qdrant, retrieve matching clauses, and record latency & cost metrics.
2. **POST `/api/compare`**: Execute regional comparisons between FSSAI, FDA, and EFSA.
3. **GET `/api/admin/metrics`**: Expose request counts, user distributions, and aggregated costs to authenticated administrator roles.
4. **GET `/api/news`**: Display the latest food safety alerts and updates.

---

## 🧪 Evaluation History & Benchmark Evolution

We establish a rigorous, local ground-truth evaluation process testing NiyamAI across 25 canonical test queries (covering exact clauses, numeric limits, semantic lookups, footnotes, amendments, and multi-document synthesis).

### 📈 Evaluation Runs & Model Transitions

#### Run 1: Raw Snippet Baseline (Baseline vs Advanced 1st Iteration)
* **Nature**: Evaluated raw vector database text chunks directly without conversational LLM synthesis.
* **LLM Judge**: `openai/gpt-oss-20b` (Zero-shot)
* **Results**:
  * **Answer Relevance**: **16.00%** (LLM judge penalized raw code-like table snippets for low natural language relevance)
  * **Faithfulness**: **70.80%** (Strict statutory overlap, but conversational flow was missing)
  * **Overall Judge Score**: **40.52%**

#### Run 2: Conversational LLM Synthesis (LangChain Structure)
* **Nature**: Switched from raw snippets to full natural language answers generated via a LangChain prompt template.
* **LLM Judge**: `openai/gpt-oss-20b`
* **Results**:
  * **Answer Relevance**: **41.60%** (**+25.60% improvement**)
  * **Faithfulness**: **48.24%** (Reflected strict hallucination audit over generated legal text)
  * **Overall Judge Score**: **41.29%**

#### Run 3: Active High-Fidelity LLM Judge (OpenAI 120B)
* **Nature**: Upgraded RAG synthesis and the evaluation judge model to Groq's active high-fidelity 120B parameter model.
* **LLM Judge**: `openai/gpt-oss-120b`
* **Results**:
  * **Answer Relevance**: **56.76%** (**+15.16% improvement**)
  * **Faithfulness**: **52.84%**
  * **Context Precision**: **50.08%**
  * **Overall Judge Score**: **53.20%**

#### Run 4 (Target Upgrade): Qdrant Native Hybrid Search + FlashRank Reranker
* **Nature**: Upgrading vector storage to Qdrant (combining `BAAI/bge-large-en-v1.5` dense embeddings + SPLADE sparse token vectors) and applying CPU-optimized **FlashRank Reranking** (`ms-marco-MiniLM-L-6-v2`) before synthesis.
* **Target Metric Goals**:
  * **Recall@5**: **90.0%+**
  * **LLM Faithfulness**: **85.0%+**
  * **Answer Relevance**: **80.0%+**


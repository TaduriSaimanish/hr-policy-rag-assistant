# HR Policy RAG Assistant (Single-Source RAG)

An end-to-end RAG system built to parse HR Employee Handbooks, execute query decomposition, perform vector retrieval with local embeddings, re-rank candidate chunks using a Cross-Encoder, and evaluate output quality using RAGAS.

---

## 1. Domain & Scope
* **Selected Domain:** HR Policy Bot (Option 6)
* **In-Scope:** Employee handbook questions concerning leave policies, probation rules, remote work terms, notice periods, and code of conduct.
* **Out-of-Scope:** Personal payroll data lookups, legal representation advice, non-HR general knowledge.

---

## 2. System Architecture

```
+-------------------------------------------------------------------------+
|                               Next.js / HTML UI                        |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                             FastAPI Backend                             |
|                                                                         |
|  [PDF Upload] -> PyPDFLoader -> Token Splitter -> FAISS Vector Index    |
|                                                                         |
|  [User Query] -> Decomposer Router (Gemini 2.5 Flash via OpenRouter)     |
|                       |                                                 |
|                       v                                                 |
|                 FAISS Retriever (MiniLM-L6-v2)                          |
|                       |                                                 |
|                       v                                                 |
|           Cross-Encoder Re-Ranker (ms-marco-MiniLM-L-6-v2)               |
|                       |                                                 |
|                       v                                                 |
|             LCEL RAG Generation Chain -> Cited Response                 |
|                       |                                                 |
|                       v                                                 |
|             RAGAS Metric Logging -> `ragas_eval_logs.jsonl`             |
+-------------------------------------------------------------------------+
```

---

## 3. Tech Stack Matrix

| Module | Technology | Rationale |
| :--- | :--- | :--- |
| **API Backend** | FastAPI | High-performance asynchronous REST endpoints. |
| **Document Loader** | PyPDFLoader | Preserves exact page numbers for UI citations. |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Zero-cost local embedding generation. |
| **Vector Index** | FAISS (`IndexFlatL2`) | Efficient local similarity searching. |
| **Re-Ranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Scores query-document pairs to fix vector mis-rankings. |
| **Orchestration** | LangChain (LCEL) | Declarative chain pipeline execution. |
| **LLM Provider** | OpenRouter (`google/gemini-2.5-flash`) | OpenAI-compatible endpoint routing to Gemini 2.5 Flash. |
| **Evaluation** | RAGAS | Log metrics (faithfulness, relevancy, precision). |

---

## 4. Re-Ranking Impact Analysis (Before vs. After)

Demonstration of Cross-Encoder re-ordering retrieved chunks for query: *"What is the notice period during probation vs full employment?"*

| Chunk ID | Page | Initial FAISS Rank | Post Re-Rank Rank | Re-Rank Score | Excerpt |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `handbook.pdf_p14_c2` | 14 | 4 | **1** | `7.8210` | "Probationary employees require 15 days notice..." |
| `handbook.pdf_p14_c3` | 14 | 1 | **2** | `6.4102` | "Full-time regular employment notice period is 60 days..." |
| `handbook.pdf_p02_c1` | 2 | 2 | **3** | `2.1054` | "General office conduct and working hours..." |
| `handbook.pdf_p08_c4` | 8 | 3 | **4** | `-1.2401` | "Annual paid leave entitlement calculation..." |

---

## 5. Local Setup & Execution Guide

### Step 1: Clone and Setup Python Environment
```bash
git clone https://github.com/your-username/hr-policy-rag-assistant.git
cd hr-policy-rag-assistant/backend

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your OpenRouter key inside `.env`:
```text
OPENROUTER_API_KEY=sk-or-v1-your-actual-api-key
```
The assistant calls `google/gemini-2.5-flash` through OpenRouter's OpenAI-compatible endpoint. Override the model or endpoint via the optional `OPENROUTER_CHAT_MODEL` / `OPENROUTER_BASE_URL` env vars.

### Step 3: Launch Backend Server
```bash
uvicorn app.main:app --reload --port 8000
```

### Step 4: Open Frontend UI
Simply open `frontend/index.html` in your web browser, upload an HR handbook PDF, and query the assistant!

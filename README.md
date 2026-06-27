# 📖 Islamic Biography RAG — Q&A System for Hadrat Maulana Ashraf Ali Thanvi

A production-grade Retrieval-Augmented Generation (RAG) pipeline for querying classical Islamic biographical texts. Built to answer questions about *Hadrat Wala Hakeem ul Ummah Moulana Ashraf Ali Thanvi* with accurate, page-cited answers — even from dense Urdu/Arabic scholarly content.

**Evaluation result: 15–16 / 21 questions answered with correct page-level retrieval.**

---

## 🧩 Tech Stack

| Layer | Technology |
|---|---|
| Framework | Next.js 14 (App Router) |
| Language | TypeScript |
| LLM | Ollama (Mistral / Qwen3 8B) |
| Embeddings | `bge-large-en-v1.5` via Ollama |
| PDF Parsing | LlamaParse |
| Retrieval | Cosine similarity + BM25 hybrid |
| Streaming | Server-Sent Events (SSE) |
| Evaluation | RAGAS (Python 3.11) |

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────┐
│                  Query Layer                    │
│  1. Embed original question (cosine similarity) │
│  2. Expand query keywords (for BM25 matching)   │
└─────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────┐
│               Dual Retrieval                    │
│  Original query  ──► findRelevantChunks()       │
│  Expanded query  ──► findRelevantChunks()       │
│                                                 │
│  Merge + Deduplicate + Sort by best score       │
└─────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────┐
│               Filtering & Ranking               │
│  • Confidence gate: bestScore >= 0.25           │
│  • Strip metadata/header chunks                 │
│  • Take top-K candidates                        │
└─────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────┐
│              Generation & Verification          │
│  • Structured [EVIDENCE N] prompt with pages    │
│  • Streaming answer via Ollama                  │
│  • Post-generation answer verification          │
│  • RAGAS evaluation logging                     │
└─────────────────────────────────────────────────┘
     │
     ▼
  Streamed Answer + Sources + Page Citations
```

---

## ✨ Key Features

- **Hybrid Retrieval** — Combines dense embeddings (cosine similarity) with sparse BM25 keyword matching. Both run in parallel and results are merged for best coverage.
- **Query Expansion** — Original question is expanded with semantically related terms before BM25 retrieval, improving recall on classical transliterated terms.
- **Confidence Gating** — Answers below a combined score threshold of `0.25` are rejected with a clean "not found" message rather than hallucinated answers.
- **Metadata Chunk Filtering** — Header/footer/page-number chunks are detected and stripped before being sent to the LLM.
- **Structured Evidence Prompting** — Each chunk is passed to the LLM as `[EVIDENCE N] Page: X Text: ...`, grounding the answer in specific pages.
- **Answer Verification** — After generation, the answer is verified against retrieved chunks to flag unsupported claims.
- **SSE Streaming** — Answer streams token-by-token to the frontend for a responsive UX.
- **RAGAS Logging** — Every Q&A pair is logged with contexts for offline RAGAS evaluation (faithfulness, context precision, answer relevancy).

---

## 🔍 Retrieval Pipeline Detail

### 1. Embedding Model
Switched from `nomic-embed-text` → `bge-large-en-v1.5` after finding that `bge-large` handles transliterated Urdu/Arabic terms significantly better for semantic similarity.

### 2. BM25 + Cosine Hybrid
```
combined_score = α × cosine_score + (1 - α) × bm25_score
```
Both retrieval paths run in parallel using `Promise.all()`. Results are deduplicated by chunk ID and sorted by `combined` score (falling back to `cosine` if BM25 is unavailable).

### 3. Chunk Design
Chunks are stored with two text fields:
- `text` — raw extracted text (used for BM25 keyword matching)
- `embedText` — cleaned, normalized text (used for embedding and LLM context)

This separation was critical: embedding the raw text polluted vector representations with HTML tags, page numbers, and header noise.

---

## 📊 Evaluation Results

Tested against 21 domain-specific questions covering events, dreams, predictions, spiritual incidents, and scholarly opinions from the biography.

| Metric | Score |
|---|---|
| Exact page match | 9 / 21 |
| Page ± 1–2 (adjacent chunk) | 6 / 21 |
| Thematically relevant but wrong section | 4 / 21 |
| Not retrieved | 2 / 21 |
| **Effective accuracy (correct + adjacent)** | **~71%** |

### Hardest Cases
- **Multi-page span questions** — Answer content split across two page boundaries
- **Semantically ambiguous queries** — Similar content appears in multiple sections of the book
- **Classical transliteration mismatches** — The same name/term spelled differently across chunks

---

## 🚧 Challenges Overcome

| Problem | Solution |
|---|---|
| Embedding model poor on Urdu/Arabic | Migrated to `bge-large-en-v1.5` |
| BM25 normalization bug causing false positives | Fixed score normalization in BM25 implementation |
| Ollama context window truncation | Explicitly set `num_ctx` parameter |
| HTML tag leakage into embeddings | Separated `embedText` from raw `text` field |
| Orphan chunks with no heading context | Added heading propagation during chunking |
| Page number offset in LlamaParse output | Applied `page - 2` correction in evidence prompt |
| Query-chunk language mismatch | Added query expansion before BM25 retrieval |

---

## 🚀 Getting Started

### Prerequisites

- Node.js 18+
- [Ollama](https://ollama.ai) running locally
- Ollama models pulled:
  ```bash
  ollama pull mistral        # or qwen3:8b
  ollama pull bge-large      # embedding model
  ```

### Installation

```bash
git clone https://github.com/your-username/islamic-biography-rag
cd islamic-biography-rag
npm install
```

### Environment Variables

```env
OLLAMA_BASE_URL=http://localhost:11434
LLAMA_PARSE_API_KEY=your_llamaparse_key
```

### Run

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000), upload the biography PDF, and start asking questions.

---

## 📁 Project Structure

```
├── app/
│   ├── api/
│   │   ├── ask/route.ts          # Main RAG pipeline (SSE streaming)
│   │   ├── upload/route.ts       # PDF ingestion + chunking
│   │   └── embed/route.ts        # Embedding generation
│   └── page.tsx                  # Chat UI
├── lib/
│   ├── retrieval.ts              # Cosine + BM25 hybrid retrieval
│   ├── embeddings.ts             # Ollama embedding wrapper
│   ├── bm25.ts                   # BM25 implementation
│   ├── chunking.ts               # PDF chunk processing
│   ├── prompts.ts                # LLM prompt templates
│   └── verification.ts           # Answer verification logic
├── eval/
│   └── ragas_eval.py             # RAGAS evaluation script
└── README.md
```

---

## 🧪 RAGAS Evaluation

Every answered question is logged in RAGAS format:

```json
{
  "question": "...",
  "answer": "...",
  "contexts": ["chunk1 text", "chunk2 text"],
  "ground_truth": ""
}
```

To run offline evaluation:

```bash
cd eval
python -m venv venv
source venv/bin/activate
pip install ragas datasets langchain-community
python ragas_eval.py
```

---

## 🌱 Future Improvements

- [ ] Multi-biography support — query any uploaded Islamic scholar biography
- [ ] Urdu/Arabic UI for end users
- [ ] Cross-encoder reranker for improved top-K precision
- [ ] PDF page deep-link from answer citations
- [ ] Cloud deployment (Vercel + RunPod for Ollama)
- [ ] Fine-tuned embedding model on Urdu transliteration pairs

---

## 👤 Author

**Shoeb** — Senior Full Stack & AI Engineer  
Hyderabad, India

[GitHub](https://github.com/creativemind1) · [LinkedIn](https://www.linkedin.com/in/mohammed-shoeb-m-5237b346/)

---

## 📄 License

MIT
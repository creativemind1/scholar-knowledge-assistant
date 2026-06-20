# Scholar RAG — Biography Intelligence

A Next.js RAG (Retrieval Augmented Generation) system that lets you upload any scholar biography PDF and ask questions about it. Powered entirely by your local Ollama models — no data leaves your machine.

## How it works

**Phase 1 — Load Biography (one time)**
1. Upload a PDF biography
2. Text is extracted using `pdf-parse`
3. Text is split into overlapping chunks (~400 words each)
4. Each chunk is embedded using `nomic-embed-text` via Ollama
5. Embeddings stored in a local JSON file (`data/chunks.json`)

**Phase 2 — Ask Questions (every query)**
1. Your question is embedded using the same model
2. Cosine similarity search finds the 4 most relevant chunks
3. Those chunks + your question are sent to Mistral as context
4. Mistral answers based ONLY on what's in the PDF
5. Answer streams back token by token

## Prerequisites

Make sure Ollama is running locally:

```bash
# Start Ollama
ollama serve

# Pull required models
ollama pull mistral
ollama pull nomic-embed-text
```

## Setup

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Configuration

Edit `.env.local` to change models:

```env
OLLAMA_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
CHAT_MODEL=mistral
```

You can swap `mistral` for `llama3`, `phi3`, `gemma2` etc — any model you have pulled in Ollama.

## Project Structure

```
scholar-rag/
├── app/
│   ├── api/
│   │   ├── upload/route.ts   # PDF processing + embedding
│   │   ├── query/route.ts    # RAG query + streaming answer
│   │   └── status/route.ts   # Check if biography is loaded
│   ├── page.tsx              # Main UI
│   └── layout.tsx
├── lib/
│   ├── ollama.ts             # Ollama API helpers
│   └── vectorStore.ts        # Embedding storage + cosine similarity
└── data/
    └── chunks.json           # Generated on first upload (gitignored)
```

## Notes

- Works with text-based PDFs only (not scanned images)
- For scanned PDFs, run OCR first (e.g. Adobe Acrobat, tesseract)
- Embedding a 200-page PDF takes ~5-10 minutes depending on your hardware
- Subsequent queries are fast (< 5 seconds typically)

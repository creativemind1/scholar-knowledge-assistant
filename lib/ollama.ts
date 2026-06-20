const OLLAMA_BASE = process.env.OLLAMA_URL || "http://localhost:11434";

export async function getEmbedding(
  text: string,
  type: "document" | "query" = "document",
): Promise<number[]> {
  const prefix = type === "query" ? "search_query: " : "search_document: ";

  const res = await fetch(`${OLLAMA_BASE}/api/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: process.env.EMBED_MODEL || "nomic-embed-text",
      prompt: prefix + text,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Ollama embeddings error: ${err}`);
  }

  const data = await res.json();
  return data.embedding;
}

export async function generateAnswer(
  prompt: string,
  onChunk: (text: string) => void,
): Promise<void> {
  const res = await fetch(`${OLLAMA_BASE}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: process.env.CHAT_MODEL || "mistral",
      prompt,
      stream: true,
      options: {
        temperature: 0.1,
      },
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Ollama generate error: ${err}`);
  }

  const reader = res.body?.getReader();
  const decoder = new TextDecoder();

  if (!reader) throw new Error("No response body");

  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (let line of lines) {
      line = line.trim();
      if (!line) continue;
      if (line.startsWith("data:")) {
        line = line.replace(/^data:\s*/, "");
        if (line === "[DONE]") continue;
      }
      if (line.startsWith("event:")) {
        line = line.replace(/^event:\s*/, "");
        continue;
      }
      try {
        const json = JSON.parse(line);
        if (typeof json === "object" && json !== null) {
          if (json.response) {
            onChunk(json.response);
            continue;
          }
          if (json.output_text) {
            onChunk(json.output_text);
            continue;
          }
          if (json.text) {
            onChunk(json.text);
            continue;
          }
          if (json.content) {
            onChunk(json.content);
            continue;
          }
        }
      } catch {
        // fall through to deliver raw text
      }
      onChunk(line);
    }
  }

  if (buffer.trim()) {
    const remainder = buffer.trim();
    try {
      const json = JSON.parse(remainder);
      if (json.response) onChunk(json.response);
      else if (json.output_text) onChunk(json.output_text);
      else if (json.text) onChunk(json.text);
      else if (json.content) onChunk(json.content);
      else onChunk(remainder);
    } catch {
      onChunk(remainder);
    }
  }
}

export async function buildRAGPrompt(
  question: string,
  contextChunks: { text: string; page: number }[],
): Promise<string> {
  const context = contextChunks
    .map((c) => `(Page ${c.page}):\n${c.text}`)
    .join("\n\n---\n\n");

  return `You are a biography assistant. Answer using ONLY the text below.

RULES:
1. Use ONLY the provided text. Do NOT use outside knowledge.
2. If answer is NOT in text, say: "The biography does not contain information about this."
3. Do NOT hallucinate or invent facts.
4. Extract exact information from text (names, places, dates).
5. Include page numbers: (page X).

TEXT:

${context}

QUESTION:
${question}

ANSWER:`;
}

// NOTE: This function is currently unused — your ask route imports
// extractEvidence from "@/lib/ollamaHelpers" instead. Keeping this fixed
// in case it's the one actually wired up, or delete it to avoid confusion
// with the ollamaHelpers version.
export async function extractEvidence(
  question: string,
  chunks: { text: string; page: number }[],
): Promise<{ found: boolean; hits: { page: number; evidence: string }[] }> {
  const context = chunks
    .map(
      (c) => `
PAGE ${c.page}

${c.text}
`,
    )
    .join("\n---\n");

  const prompt = `
You are a document evidence extractor.

Answer ONLY from the document.

Question:
${question}

Document:
${context}

Rules:
- Find exact evidence from the document.
- Do not use your own knowledge.
- Do not explain.
- If the document does not contain the answer, reply exactly:
NOT_FOUND

Return JSON only:

{
 "found": true,
 "hits": [
   {
     "page": 12,
     "evidence": "exact text from document"
   }
 ]
}

If not found:

{
 "found": false,
 "hits":[]
}
`;

  const res = await fetch(`${OLLAMA_BASE}/api/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: process.env.CHAT_MODEL || "mistral",
      prompt,
      stream: false,
      format: "json",
      options: {
        temperature: 0,
      },
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Ollama generate error: ${err}`);
  }

  const data = await res.json();
  const cleaned = (data.response as string).replace(/```json|```/g, "").trim();

  try {
    return JSON.parse(cleaned);
  } catch {
    throw new Error("Could not parse evidence extractor response");
  }
}

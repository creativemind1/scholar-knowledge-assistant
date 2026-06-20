const OLLAMA_BASE = process.env.OLLAMA_URL || "http://localhost:11434";

// Strip markdown code fences some models wrap JSON output in,
// even when format: "json" is requested.
function stripJsonFences(text: string): string {
  return text
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();
}

function extractJsonSubstring(text: string): string | null {
  const m = text.match(/\{[\s\S]*\}/);
  return m ? m[0] : null;
}

// Rewrite the user's question as a concise retrieval query.
export async function rewriteQuestion(question: string): Promise<string> {
  const prompt = `Rewrite the user question as a concise retrieval query for a biography document.\n\nQuestion:\n${question}\n\nRules:\n- Keep meaning exactly.\n- Use short, factual retrieval language.\n- Do not add any explanation.\n- Return only the rewritten query.\n`;

  const res = await fetch(`${OLLAMA_BASE}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: process.env.CHAT_MODEL || "mistral",
      prompt,
      stream: false,
      options: { temperature: 0 },
    }),
  });

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`rewriteQuestion failed: ${txt}`);
  }

  const data = await res.json();
  return data.response.trim();
}

export async function rerankCandidates(
  question: string,
  candidates: { id: string; text: string }[],
): Promise<Record<string, number>> {
  if (!candidates || candidates.length === 0) return {};
  const items = candidates
    .map((c) => `ITEM_ID:${c.id}\nTEXT:${c.text.replace(/\n/g, " ")}`)
    .join("\n\n---\n\n");

  const prompt = `You are a scorer that determines whether a text contains direct evidence to answer the question.\n\nRespond ONLY with a JSON object mapping item ids to a numeric score between 0 and 1.\n\nQuestion:\n${question}\n\nCandidates:\n${items}\n\nRules:\n- Do not add any extra commentary.\n- Output valid JSON only.\n- Example: {"item1":0.83,"item2":0.12}\n`;

  const res = await fetch(`${OLLAMA_BASE}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: process.env.CHAT_MODEL || "mistral",
      prompt,
      stream: false,
      format: "json",
      options: { temperature: 0 },
    }),
  });

  if (!res.ok) {
    const txt = await res.text();
    console.error("rerankCandidates error:", txt);
    throw new Error("Rerank request failed");
  }

  const data = await res.json();
  const cleaned = stripJsonFences(data.response ?? "");

  try {
    return JSON.parse(cleaned);
  } catch {
    const sub = extractJsonSubstring(cleaned);
    if (sub) {
      try {
        return JSON.parse(sub);
      } catch {
        // fall through
      }
    }

    throw new Error("Could not parse reranker JSON response");
  }
}

// extractEvidence returns structured JSON:
// { found: boolean, hits?: [{chunk_id,page,evidence}] }
export async function selectRelevantChunks(
  question: string,
  chunks: { id?: string; text: string; page: number }[],
  subjectName: string,
): Promise<{ found: boolean; chunk_ids: string[] }> {
  const items = chunks
    .map(
      (c) =>
        `CHUNK_ID:${c.id || "unknown"}\nPAGE:${c.page}\nTEXT:${c.text.replace(/\n/g, " ")}`,
    )
    .join("\n\n---\n\n");

  const prompt = `You are selecting which document chunks are relevant to a question.

This document is a biography of ${subjectName}. It also contains parables, hadith, and unrelated anecdotes that share keywords with the question but aren't relevant.

Question:
${question}

Chunks:
${items}

Rules:
- Identify EVERY chunk that is genuinely relevant to answering the question.
- A chunk is relevant if it directly addresses the question topic — do not select a chunk just because it shares a keyword.
- Do NOT quote, rewrite, or summarize any text. Only return chunk IDs.
- Return JSON exactly in this format:
  {"found":true,"chunk_ids":["chunk_10","chunk_15"]}
- If no chunk is relevant, return exactly: {"found":false,"chunk_ids":[]}
- Do NOT add any other text or explanation.
`;

  const res = await fetch(`${OLLAMA_BASE}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: process.env.CHAT_MODEL || "mistral",
      prompt,
      stream: false,
      format: "json",
      options: { temperature: 0 },
    }),
  });

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`selectRelevantChunks failed: ${txt}`);
  }

  const data = await res.json();
  const cleaned = stripJsonFences(data.response ?? "");
  try {
    const parsed = JSON.parse(cleaned);
    if (
      parsed &&
      typeof parsed.found === "boolean" &&
      Array.isArray(parsed.chunk_ids)
    ) {
      return parsed;
    }
    throw new Error("Invalid selector output");
  } catch {
    const sub = extractJsonSubstring(cleaned);
    if (sub) {
      try {
        const parsed = JSON.parse(sub);
        if (
          parsed &&
          typeof parsed.found === "boolean" &&
          Array.isArray(parsed.chunk_ids)
        ) {
          return parsed;
        }
      } catch {
        // fall through
      }
    }
    throw new Error("Could not parse selector JSON response");
  }
}

// Very small verification: ensure named entities or numbers in answer exist in evidence
export function verifyAnswer(
  answer: string,
  hits: { chunk_id?: string; page: number; evidence: string }[],
): { verified: boolean; missing?: string[] } {
  const evidenceCorpus = hits.map((h) => h.evidence).join(" \n ");
  const missing: string[] = [];

  // find capitalized sequences (naive NER) and numbers/dates
  const named = Array.from(
    new Set(answer.match(/\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b/g) || []),
  );
  for (const n of named) {
    if (n.length > 2 && !evidenceCorpus.includes(n)) {
      missing.push(n);
    }
  }
  const nums = Array.from(new Set(answer.match(/\b\d{2,4}\b/g) || []));
  for (const num of nums) if (!evidenceCorpus.includes(num)) missing.push(num);

  return { verified: missing.length === 0, missing };
}

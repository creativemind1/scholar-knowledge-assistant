import fs from "fs";
import path from "path";
import { expandQuery } from "./synonyms";



const DB_PATH = path.join(process.cwd(), "data", "chunks.json");

export interface Chunk {
  id: string;
  text: string;
  embedding: number[];
  page: number;
  scholarName: string;
  heading: string;
  embedText: string;
}

export interface DB {
  chunks: Chunk[];
  scholarName: string;
  uploadedAt: string;
}

export function isMetadataChunk(text: string) {
  const lower = text.toLowerCase();

  const patterns = [
    "table of contents",
    "contents",
    "chapter one",
    "chapter two",
    "chapter three",
    "chapter four",
    "chapter five",
    "chapter six",
  ];

  const hasChapter = /chapter\s+(one|two|three|four|five|six|\d+)/i.test(text);

  const hasManyPageNumbers =
    (text.match(/\b\d+\b/g) || []).length > 5;

  return (
    patterns.some(p => lower.includes(p)) ||
    (hasChapter && hasManyPageNumbers)
  );
}

export function readDB(): DB {
  if (!fs.existsSync(DB_PATH)) {
    return { chunks: [], scholarName: "", uploadedAt: "" };
  }
  const raw = fs.readFileSync(DB_PATH, "utf-8");
  return JSON.parse(raw);
}

function writeDB(db: DB) {
  fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });
  fs.writeFileSync(DB_PATH, JSON.stringify(db, null, 2));
}

export function normalizeText(text: string): string {
  return text
    .replace(/[^\x20-\x7E\n\r\t]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function saveChunks(chunks: Chunk[], scholarName: string) {
  writeDB({ chunks, scholarName, uploadedAt: new Date().toISOString() });
}

export function getDB(): DB {
  return readDB();
}

export function hasData(): boolean {
  const db = readDB();
  return db.chunks.length > 0;
}

// Cosine similarity between two vectors
export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Simple tokenization for BM25
export function tokenize(text: string) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

// Compute idf map across docs
function computeIdf(docs: string[]) {
  const df: Record<string, number> = {};
  const N = docs.length;
  for (const d of docs) {
    const seen = new Set<string>();
    for (const t of tokenize(d)) {
      if (!seen.has(t)) {
        df[t] = (df[t] || 0) + 1;
        seen.add(t);
      }
    }
  }
  const idf: Record<string, number> = {};
  for (const [t, cnt] of Object.entries(df)) {
    idf[t] = Math.log(1 + (N - cnt + 0.5) / (cnt + 0.5));
  }
  return idf;
}

function bm25Score(
  query: string,
  doc: string,
  idf: Record<string, number>,
  avgdl: number,
) {
  const k1 = 1.2;
  const b = 0.75;
  const tokens = tokenize(doc);
  const docLen = tokens.length;
  const freqs: Record<string, number> = {};
  for (const t of tokens) freqs[t] = (freqs[t] || 0) + 1;
  let score = 0;
  for (const qtok of tokenize(query)) {
    const idfv = idf[qtok] || 0;
    const f = freqs[qtok] || 0;
    const denom = f + k1 * (1 - b + (b * docLen) / avgdl);
    score += idfv * ((f * (k1 + 1)) / (denom || 1));
  }
  return score;
}

export type RelevanceHit = {
  chunk: Chunk;
  cosine: number;
  bm25: number;
  combined: number;
  rerank?: number; // cross-encoder score, if you wire rerankCandidates back in later
};

// Hybrid retrieval: cosine (semantic) + BM25 (keyword), min-max normalized
// per query so the two scales are comparable before blending.
export async function findRelevantChunks(
  queryEmbedding: number[],
  question: string,
): Promise<RelevanceHit[]> {
  const db = readDB();

  if (db.chunks.length === 0) {
    return [];
  }
  // const expandedQuestion = await expandQuery(question);
  const expandedQuestion = question; // skip expandQuery entirely for now


  console.log("question:", expandedQuestion);

  const idf = computeIdf(db.chunks.map((c) => c.text));
  const avgdl =
    db.chunks.reduce((sum, c) => sum + tokenize(c.text).length, 0) /
    db.chunks.length;

  const raw = db.chunks.map((chunk) => {
    const cosine = cosineSimilarity(queryEmbedding, chunk.embedding);
    const bm25 = bm25Score(expandedQuestion, chunk.embedText, idf, avgdl);
    return { chunk, cosine, bm25 };
  });

  // Normalize each score to 0-1 across this query's candidate set
  const cosVals = raw.map((r) => r.cosine);
  const bm25Vals = raw.map((r) => r.bm25);
  const minMax = (vals: number[]) => {
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    return { min, range: max - min || 1 };
  };
  const cosStats = minMax(cosVals);
  const bm25Stats = minMax(bm25Vals);

  const results: RelevanceHit[] = raw.map((r) => {
    const normCos = (r.cosine - cosStats.min) / cosStats.range;
    const normBm25 =
      r.bm25 > 0.1 ? (r.bm25 - bm25Stats.min) / bm25Stats.range : 0;    // Weight semantic similarity higher, but let keyword matches pull
    // exact names/dates/places up the ranking.
    const combined = normCos * 0.70 + normBm25 * 0.30;
    return { chunk: r.chunk, cosine: r.cosine, bm25: r.bm25, combined };
  });

  results.sort((a, b) => b.combined - a.combined);

  return results.slice(0, 20);
}

// Given a set of selected chunks, pull in their immediate neighbors
// (by original chunking order) so a story that was split across chunk
// boundaries arrives whole, instead of with a gap in the middle.
export function expandWithNeighbors(
  selectedChunks: Chunk[],
  windowSize: number = 1,
): Chunk[] {
  const db = readDB();
  if (db.chunks.length === 0) return selectedChunks;

  // Chunks were created sequentially as chunk_0, chunk_1, ... in document
  // order, so their position in db.chunks IS their narrative order.
  const indexById = new Map<string, number>();
  db.chunks.forEach((c, idx) => indexById.set(c.id, idx));

  const expandedIndices = new Set<number>();
  for (const c of selectedChunks) {
    const idx = indexById.get(c.id);
    if (idx === undefined) continue;
    for (let offset = -windowSize; offset <= windowSize; offset++) {
      const neighborIdx = idx + offset;
      if (neighborIdx >= 0 && neighborIdx < db.chunks.length) {
        expandedIndices.add(neighborIdx);
      }
    }
  }

  // Sort by original document order so the merged text reads as a
  // continuous narrative, not in relevance-score order.
  const sortedIndices = Array.from(expandedIndices).sort((a, b) => a - b);
  return sortedIndices.map((idx) => db.chunks[idx]);
}

import fs from "fs";
import path from "path";
import { expandQuery } from "./ollama";



const DB_PATH = path.join(process.cwd(), "data", "chunks.json");

export interface Chunk {
  id: string;
  text: string;
  embedding: number[];
  page: number;
  scholarName: string;
  heading: string;
  embedText: string;
  metadata?: {
    sectionTitle?: string;
    wordCount?: number;
    entities?: string[];
    containsHardoi?: boolean;
    containsSpring?: boolean;
    [key: string]: any; // For future flexibility
  };
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

const STOPWORDS = new Set([
  // Common English
  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
  'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been',
  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
  'should', 'may', 'might', 'shall', 'can', 'this', 'that', 'these',
  'those', 'it', 'its', 'also', 'just', 'more', 'some', 'such', 'said',
  'says', 'who', 'what', 'where', 'when', 'how', 'why', 'which', 'about',
  'their', 'they', 'them', 'then', 'than', 'there', 'here', 'each',
  'all', 'any', 'both', 'few', 'not', 'only', 'own', 'same', 'so',
  'very', 'as', 'if', 'his', 'her', 'him', 'our', 'your', 'my', 'we',
  'he', 'she', 'you', 'me', 'us', 'i',
  // Generic words causing false BM25 matches in your RAG
  'rules', 'book', 'law', 'laws', 'contains', 'contain', 'tell',
  'describe', 'explain', 'mention', 'according', 'based', 'related',
  'information', 'regarding', 'about', 'following', 'said', 'says',
  'one', 'two', 'three', 'like', 'see', 'now', 'get', 'go', 'come',
  'know', 'think', 'make', 'take', 'give', 'use', 'find', 'want'
])

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(t => t.length > 2 && !STOPWORDS.has(t))
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

  const idf = computeIdf(db.chunks.map((c) => c.embedText));
  const avgdl =
    db.chunks.reduce((sum, c) => sum + tokenize(c.embedText).length, 0) /
    db.chunks.length;

  const raw = db.chunks.map((chunk) => {
    const cosine = cosineSimilarity(queryEmbedding, chunk.embedding);
    const bm25 = bm25Score(question, chunk.embedText, idf, avgdl);
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
    const combined = normCos * 0.75 + normBm25 * 0.25 // was 0.70/0.30
    //const combined = r.cosine;
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


export function extractEntities(text: string): string[] {
  // Look for capitalized words that might be names or places
  const entities = text.match(/[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/g) || [];
  return [...new Set(entities)].slice(0, 10);
}

export function expandChunks(
  topChunks: Chunk[],
  allChunks: Chunk[],
  windowSize: number = 5,  // Increased default
  maxChunks: number = 30
): Chunk[] {
  const expanded: Chunk[] = [];
  const seen = new Set<string>();

  for (const chunk of topChunks) {
    const index = allChunks.findIndex(c => c.id === chunk.id);

    if (index === -1) continue;

    // Get chunks BEFORE with window size
    for (let i = 1; i <= windowSize; i++) {
      const prevIndex = index - i;
      if (prevIndex >= 0) {
        const prev = allChunks[prevIndex];
        if (!seen.has(prev.id)) {
          expanded.push(prev);
          seen.add(prev.id);
        }
      }
    }

    // Current chunk
    if (!seen.has(chunk.id)) {
      expanded.push(chunk);
      seen.add(chunk.id);
    }

    // Get chunks AFTER with window size
    for (let i = 1; i <= windowSize; i++) {
      const nextIndex = index + i;
      if (nextIndex < allChunks.length) {
        const next = allChunks[nextIndex];
        if (!seen.has(next.id)) {
          expanded.push(next);
          seen.add(next.id);
        }
      }
    }
  }

  // Sort by page number
  const sorted = expanded.sort((a, b) => a.page - b.page);

  // Limit to maxChunks
  return sorted.slice(0, maxChunks);
}
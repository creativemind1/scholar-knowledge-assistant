import {
  NextRequest,
  NextResponse
} from "next/server";
import {
  getEmbedding
} from "@/lib/ollama";
import {
  saveChunks,
  Chunk,
  extractEntities
} from "@/lib/vectorStore";
import llamaParseJson from "@/llamaParseJson.json";
import { buildRAGChunksFromItems } from "@/lib/build-chunks-from-items";

type ParsedPage = { page_number: number; text: string };
type ParsedDocument = { text: { pages: ParsedPage[] } };
interface RAGChunk {
  id: string;
  page: number;
  title?: string;
  chapter?: string;
  section?: string;
  subsection?: string;
  text: string;
  heading: string;
  embedText: string;  // pure content only, for the embedding model
}
interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
  confidence: number;
  start_index: number;
  end_index: number;
  label: string;
}

interface HeadingItem {
  type: "heading";
  md: string;
  level: number;
  value: string;
  bbox?: BoundingBox[];
}

interface TextItem {
  type: "text";
  md: string;
  value: string;
  bbox?: BoundingBox[];
}

interface LinkItem {
  type: "link";
  md: string;
  url: string;
  text: string;
  bbox?: BoundingBox[];
}

type LlamaParseItem =
  | HeadingItem
  | TextItem
  | LinkItem;

interface LlamaParsePage {
  page_number: number;
  items: LlamaParseItem[];
  page_width: number;
  page_height: number;
  success: boolean;
}


function cleanMeta(value: string) {
  return value?.trim()
    ? value
    : "N/A";

}

function stripMarkup(text: string): string {
  return text
    .replace(/<[^>]+>/g, "") // remove all HTML-like tags
    .replace(/\s+/g, " ")    // collapse extra whitespace left behind
    .trim();
}

function splitPageBySections(pageText: string): string[] {
  // Make the newline optional: use /(?:^|\n) instead of /\n
  const sectionPattern = /(?:^|\n)(\d+\.\s+[A-Z\s]+\(?[A-Z]*\)?)/g;
  const matches = [...pageText.matchAll(sectionPattern)];

  console.log(`🔍 Found ${matches.length} section headers`);
  matches.forEach(m => console.log(`   Match: "${m[1]}"`));

  if (matches.length === 0) {
    return [pageText];
  }

  const sections = [];

  for (let i = 0; i < matches.length; i++) {
    const start = matches[i].index;
    const end = i < matches.length - 1 ? matches[i + 1].index : pageText.length;
    const sectionText = pageText.substring(start, end).trim();

    if (sectionText) {
      sections.push(sectionText);
    }
  }

  return sections;
}

function extractSectionTitle(section: string): string {
  const match = section.match(/^\d+\.\s+([A-Z\s]+\(?[A-Z]*\)?)/);
  return match ? match[1].trim() : "";
}

function buildRAGChunks(pages: ParsedPage[]): RAGChunk[] {
  const chunks: RAGChunk[] = [];
  let chunkCounter = 0;

  for (const page of pages) {
    if (page.page_number < 4 || page.page_number === 7 || page.page_number === 8) continue;

    // ⭐ DEBUG: Check raw content before stripMarkup
    if (page.page_number === 159) {
      console.log('🔍 RAW CONTENT (before stripMarkup):');
      console.log(page.text.slice(0, 500));
      console.log('---');
    }

    const content = stripMarkup(page.text.trim());

    // ⭐ DEBUG: Check content after stripMarkup
    if (page.page_number === 159) {
      console.log('🔍 AFTER stripMarkup:');
      console.log(content.slice(0, 500));
      console.log('---');
    }

    if (!content) continue;

    const sections = splitPageBySections(content);

    if (page.page_number === 159) {
      console.log(`📊 After split: ${sections.length} sections`);
    }

    for (const section of sections) {
      chunks.push({
        id: `chunk_${chunkCounter++}`,
        page: page.page_number,
        text: section,
        embedText: section,
        heading: extractSectionTitle(section),
      });
    }
  }

  return chunks;
}



export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('json') as File;
    const scholarName = (formData.get('scholarName') as string) || 'Unknown Scholar';

    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    const text = await file.text();
    const parsed = JSON.parse(text);

    const pages = parsed.items.pages;
    if (!pages || !Array.isArray(pages)) {
      return NextResponse.json({ error: 'Invalid format: no pages array' }, { status: 400 });
    }

    // ⭐ Check if this is items-based or old format
    const firstPage = pages[0];
    const isItemsBased = firstPage && Array.isArray(firstPage.items);

    let textChunks: any[];
    if (isItemsBased) {
      console.log('📄 Using items-based chunking (NEW)');
      textChunks = buildRAGChunksFromItems(pages);
    } else {
      console.log('📄 Using legacy page-based chunking (OLD)');
      textChunks = buildRAGChunks(pages);
    }

    const numpages = pages[pages.length - 1].page_number;

    const encoder = new TextEncoder();
    const stream = new TransformStream();
    const writer = stream.writable.getWriter();
    const sendProgress = async (msg: object) => {
      await writer.write(encoder.encode(JSON.stringify(msg) + "\n"));
    };
    (async () => {
      try {
        await sendProgress({
          type: "start",
          total: textChunks.length,
          scholarName
        });
        const chunks: Chunk[] = [];
        for (let i = 0; i < textChunks.length; i++) {
          const chunk = textChunks[i];
          await sendProgress({
            type: "progress",
            current: i + 1,
            total: textChunks.length,
            message: `Embedding chunk ${i + 1}`
          });
          const embedding = await getEmbedding(chunk.embedText, "document");
          chunks.push({
            id: `chunk_${i}`,
            text: chunk.text,
            heading: chunk.heading,
            embedding,
            page: chunk.page,
            scholarName,
            embedText: chunk.embedText,
            // ⭐ NEW: Store metadata for better retrieval
            metadata: chunk.metadata || {
              sectionTitle: chunk.heading || 'Untitled',
              wordCount: chunk.text.split(/\s+/).length,
              entities: extractEntities(chunk.text)
            }
          });
        }
        saveChunks(chunks, scholarName);
        await sendProgress({
          type: "done",
          totalChunks: chunks.length,
          pages: numpages,
          scholarName
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unknown error";
        await sendProgress({
          type: "error",
          message
        });
      } finally {
        await writer.close();
      }
    })();
    return new Response(stream.readable, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({
      error: message
    }, {
      status: 500
    });
  }
}
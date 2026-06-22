import {
  NextRequest,
  NextResponse
} from "next/server";
import {
  getEmbedding
} from "@/lib/ollama";
import {
  saveChunks,
  Chunk
} from "@/lib/vectorStore";
import llamaParseJson from "@/llamaParseJson.json";

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

function buildRAGChunks(pages: ParsedPage[]): RAGChunk[] {
  const chunks: RAGChunk[] = [];
  let chunkCounter = 0;

  for (const page of pages) {
    if (page.page_number < 4 || page.page_number === 7 || page.page_number === 8) continue;
    const content = stripMarkup(page.text.trim());
    if (!content) continue;

    // Agar page chota hai (jaisa biography text normally hota hai), 
    // pura page hi ek chunk bana do
    const words = content.split(/\s+/).filter(Boolean);

    if (words.length <= 300) {
      // Pura page = ek chunk, page number 100% accurate rahega
      chunks.push({
        id: `chunk_${chunkCounter++}`,
        page: page.page_number,
        text: content,
        embedText: content,
        heading: "", // ya heading-detect logic alag se
      });
    } else {
      // Bada page hai to andar hi split karo, lekin page number same rahega
      for (let start = 0; start < words.length; start += 200) {
        const chunkWords = words.slice(start, start + 250);
        chunks.push({
          id: `chunk_${chunkCounter++}`,
          page: page.page_number, // YE NEVER GALAT HOGA — single page se hi aaya
          text: chunkWords.join(" "),
          embedText: chunkWords.join(" "),
          heading: "",
        });
      }
    }
  }

  return chunks;
}

export async function POST(req: NextRequest) {
  try {

    const contentType = req.headers.get("content-type") || "";
    let parsed: ParsedDocument;
    let scholarName = "Unknown Scholar";

    if (contentType.includes("application/json")) {
      const body = await req.json();
      parsed = body as ParsedDocument;
      scholarName = (body.scholarName as string) || scholarName;
    } else {
      // Accept a JSON file under "json" via multipart form data
      const formData = await req.formData();
      const file = formData.get("json") as File;
      scholarName = (formData.get("scholarName") as string) || scholarName;
      if (!file) {
        return NextResponse.json({
          error: "No parsed-document JSON provided"
        }, {
          status: 400
        });
      }
      const text = await file.text();
      parsed = JSON.parse(text) as ParsedDocument;
    }

    const pages = parsed?.pages;
    if (!Array.isArray(pages) || pages.length === 0) {
      return NextResponse.json({
        error: "Parsed document has no pages"
      }, {
        status: 400
      });
    }

    const textChunks =
      buildRAGChunks(
        parsed.pages,
        250,
        50
      );

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
            embedText: chunk.embedText
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
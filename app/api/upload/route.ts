import { NextRequest, NextResponse } from "next/server";
import { getEmbedding } from "@/lib/ollama";
import { saveChunks, Chunk } from "@/lib/vectorStore";

// Paragraph-aware chunking.
//
// Keep chunks SMALL (maxWords ~250). Small chunks = precise, well-separated
// embeddings = accurate retrieval ranking. Narrative completeness (stories
// that span multiple chunks) is handled at QUERY time by
// expandWithNeighbors() in the ask route, not by making chunks huge.
// Large chunks (e.g. 3000 words) dilute each embedding across many
// unrelated topics and badly hurt retrieval precision.
function chunkText(text: string, maxWords = 250, overlapWords = 50): string[] {
  const paragraphs = text
    .replace(/\r\n/g, "\n")
    .split(/\n\s*\n+/)
    .map((p) => p.trim())
    .filter(Boolean);

  const chunks: string[] = [];
  let bufferWords: string[] = [];

  function flushBuffer() {
    if (bufferWords.length === 0) return;
    const chunk = bufferWords.join(" ").trim();
    if (chunk.length > 0) chunks.push(chunk);
    if (bufferWords.length > overlapWords) {
      bufferWords = bufferWords.slice(bufferWords.length - overlapWords);
    } else {
      bufferWords = [];
    }
  }

  for (const p of paragraphs) {
    const words = p.split(/\s+/).filter(Boolean);

    if (words.length > maxWords) {
      const sentences = p.split(/(?<=[.!?])\s+/);
      for (const s of sentences) {
        const sw = s.split(/\s+/).filter(Boolean);
        if (bufferWords.length + sw.length > maxWords) {
          flushBuffer();
        }
        bufferWords.push(...sw);
      }
    } else {
      if (bufferWords.length + words.length > maxWords) {
        flushBuffer();
      }
      bufferWords.push(...words);
    }
  }

  flushBuffer();

  // Merge any short trailing fragment into the previous chunk rather than
  // discarding it or shipping it as an orphan with no surrounding context.
  const merged: string[] = [];
  for (const c of chunks) {
    if (c.length < 50 && merged.length > 0) {
      merged[merged.length - 1] += " " + c;
    } else {
      merged.push(c);
    }
  }
  return merged;
}

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("pdf") as File;
    const scholarName =
      (formData.get("scholarName") as string) || "Unknown Scholar";

    if (!file) {
      return NextResponse.json(
        { error: "No PDF file provided" },
        { status: 400 },
      );
    }

    if (!file.name.endsWith(".pdf")) {
      return NextResponse.json(
        { error: "Only PDF files are supported" },
        { status: 400 },
      );
    }

    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const pdfParseModule = (await import("pdf-parse")) as any;
    const pdfParse = pdfParseModule.default || pdfParseModule;
    const pdfData = await pdfParse(buffer);

    const rawText = pdfData.text;

    if (!rawText || rawText.trim().length < 100) {
      return NextResponse.json(
        {
          error:
            "Could not extract text from PDF. Make sure it is not a scanned image PDF.",
        },
        { status: 400 },
      );
    }

    // IMPORTANT: keep these small (250/50). See note above chunkText().
    const textChunks = chunkText(rawText, 250, 50);
    if (textChunks.length === 0) {
      return NextResponse.json(
        { error: "No text content found in PDF" },
        { status: 400 },
      );
    }

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
          scholarName,
        });

        const chunks: Chunk[] = [];

        for (let i = 0; i < textChunks.length; i++) {
          const text = textChunks[i];

          await sendProgress({
            type: "progress",
            current: i + 1,
            total: textChunks.length,
            message: `Embedding chunk ${i + 1} of ${textChunks.length}...`,
          });

          // "document" prefix required for nomic-embed-text's asymmetric
          // embedding space — see "query" prefix used in the ask route.
          const embedding = await getEmbedding(text, "document");

          chunks.push({
            id: `chunk_${i}`,
            text,
            embedding,
            // NOTE: still a proportional estimate, not the chunk's real
            // PDF page. Known limitation — switch to pdfjs-dist with
            // position-aware extraction to get true page numbers.
            page: Math.floor((i / textChunks.length) * pdfData.numpages) + 1,
            scholarName,
          });
        }

        saveChunks(chunks, scholarName);

        await sendProgress({
          type: "done",
          totalChunks: chunks.length,
          pages: pdfData.numpages,
          scholarName,
        });
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "Unknown error";
        await sendProgress({ type: "error", message });
      } finally {
        await writer.close();
      }
    })();

    return new Response(stream.readable, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

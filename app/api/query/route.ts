import { NextRequest } from "next/server";
import { getEmbedding, generateAnswer } from "@/lib/ollama";
import { rerankDocuments, verifyAnswer } from "@/lib/ollamaHelpers";
import {
  findRelevantChunks,
  getDB,
  hasData,
  expandWithNeighbors,
  isMetadataChunk,
} from "@/lib/vectorStore";

export async function POST(req: NextRequest) {
  const db = getDB();
  try {
    const { question } = await req.json();

    if (!question?.trim()) {
      return new Response(JSON.stringify({ error: "Question is required" }), {
        status: 400,
      });
    }

    if (!hasData()) {
      return new Response(
        JSON.stringify({
          error: "No biography loaded. Please upload a PDF first.",
        }),
        { status: 400 },
      );
    }

    const encoder = new TextEncoder();
    const stream = new TransformStream();
    const writer = stream.writable.getWriter();

    const send = async (msg: object) => {
      await writer.write(encoder.encode(JSON.stringify(msg) + "\n"));
    };

    (async () => {
      try {
        await send({ type: "status", message: "Processing question..." });
        const questionEmbedding = await getEmbedding(question, "query");

        const results = await findRelevantChunks(questionEmbedding, question);

        const bestScore = results[0]?.combined ?? results[0]?.cosine ?? 0;

        if (bestScore < 0.45 || results.length === 0) {
          await send({
            type: "answer",
            text: "The biography does not contain information about this.",
          });
          await send({ type: "done" });
          return;
        }

        const relevantChunks = results.map((r) => r.chunk).filter((c) => {
          if (isMetadataChunk(c.text)) {
            return false;
          }

          return true;
        });

        // const candidates = relevantChunks.slice(0, 20);

        // const rerankedChunks = await rerankDocuments(
        //   question,
        //   candidates
        // );



        const topChunks = relevantChunks.slice(0, 5);

        // Pull in neighboring chunks so a story split across chunk
        // boundaries arrives complete instead of with a gap.
        // const expandedChunks = expandWithNeighbors(topChunks, 1);

        const hits = topChunks.map((c) => ({
          chunk_id: c.id,
          page: c.page,
          evidence: c.embedText,
        }));

        if (hits.length === 0) {
          await send({
            type: "answer",
            text: "The biography does not contain information about this.",
          });
          await send({ type: "done" });
          return;
        }

        const sources = hits.map((c) => ({
          text:
            c.evidence.slice(0, 200) + (c.evidence.length > 200 ? "..." : ""),
          page: c.page,
        }));

        await send({ type: "sources", sources });

        const evidenceText = hits
          .map((h) => `(Page ${h.page}):\n${h.evidence}`)
          .join("\n\n---\n\n");


        console.log(evidenceText, '=======evidenceText=======')
        // STRICT-GROUNDING PROMPT. Earlier permissive rules ("always
        // answer", "only refuse if absolutely no related context") caused
        // the model to fabricate entire fictional incidents (a fake
        // Ikhlaas/movie parable, a fake Mandela meeting) when retrieval
        // was weak. This version keeps synonym tolerance (movie/cinema)
        // but removes the license to invent.
        const start = Date.now();

        const prompt = `
       [SYSTEM]
You are a warm, knowledgeable research assistant specializing in Islamic biography. Your tone is respectful, clear, and conversational—like someone sharing a fascinating fact about a beloved scholar.

**CRITICAL RULES:**
1. **Grounding:** Your answer MUST come strictly from the provided evidence chunks. Do NOT use outside knowledge.
2. **Synthesis:** Read ALL 5 chunks. The answer may be in one chunk or scattered across multiple. The chunks are NOT ranked by importance—the best chunk might be at the bottom.
3. **Natural Language:** Respond as if you're telling someone the answer in person. 
   - ✅ *"Moulana named his first child Maryam..."*
   - ❌ *"Based on Chunk 2, Page 102, the evidence states..."*
4. **Context, Not Citation:** You may briefly explain the *reason* behind the name if the evidence provides it (e.g., "He chose the name Maryam because..."), but do not cite chunk numbers, page numbers, or use phrases like "the relevant text states."
5. **Flow:** If the answer is a simple fact, give a clear, direct sentence. If it's a story, narrate it naturally with a beginning, middle, and end.
6. **Negative:** ONLY say "I couldn't find that information in the provided documents." if NONE of the chunks contain the answer, even after reading all of them carefully.

---

[USER]
**QUESTION:**
${question}

**EVIDENCE:**
${evidenceText}

---

[ASSISTANT]
[Provide your answer here in natural, flowing language. No citations. No chunk references. Just the answer, delivered like a helpful human.]
       `;

        let fullAnswer = "";
        await generateAnswer(prompt, async (chunk) => {
          fullAnswer += chunk;
          await send({ type: "chunk", text: chunk });
        });

        if (!fullAnswer.trim()) {
          fullAnswer = "The biography does not contain information about this.";
          await send({ type: "chunk", text: fullAnswer });
        }

        const verification = verifyAnswer(fullAnswer, hits);
        await send({
          type: "verification",
          verified: verification.verified,
          ...(verification.missing ? { missing: verification.missing } : {}),
        });

        await send({ type: "done" });

        //const answer = await generate(prompt);
        console.log("LLM::::::::::", Date.now() - start);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "Unknown error";
        await send({ type: "error", message });
      } finally {
        await writer.close();
      }
    })();

    return new Response(stream.readable, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
      },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return new Response(JSON.stringify({ error: message }), { status: 500 });
  }
}

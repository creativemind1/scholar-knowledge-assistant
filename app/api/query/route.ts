import { NextRequest } from "next/server";
import { getEmbedding, generateAnswer } from "@/lib/ollama";
import { verifyAnswer } from "@/lib/ollamaHelpers";
import {
  findRelevantChunks,
  getDB,
  hasData,
  expandWithNeighbors,
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
        results.slice(0, 20).forEach((r, i) => {
          console.log(
            `${i + 1}. Page ${r.chunk.page} | Combined=${r.combined.toFixed(3)}`
          );
        });
        const bestScore = results[0]?.combined ?? results[0]?.cosine ?? 0;



        if (bestScore < 0.45 || results.length === 0) {
          await send({
            type: "answer",
            text: "The biography does not contain information about this.",
          });
          await send({ type: "done" });
          return;
        }

        const relevantChunks = results.map((r) => r.chunk);
        const topChunks = relevantChunks.slice(0, 3);

        // Pull in neighboring chunks so a story split across chunk
        // boundaries arrives complete instead of with a gap.
        const expandedChunks = expandWithNeighbors(topChunks, 1);

        const hits = expandedChunks.map((c) => ({
          chunk_id: c.id,
          page: c.page,
          evidence: c.text,
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



        // STRICT-GROUNDING PROMPT. Earlier permissive rules ("always
        // answer", "only refuse if absolutely no related context") caused
        // the model to fabricate entire fictional incidents (a fake
        // Ikhlaas/movie parable, a fake Mandela meeting) when retrieval
        // was weak. This version keeps synonym tolerance (movie/cinema)
        // but removes the license to invent.
        const prompt = `You are a strict, source-grounded QA assistant. 

### CRITICAL RULES
1. **GROUNDING**: ONLY use the provided <context>. IGNORE all your internal, pre-training knowledge. If the context contradicts common sense, trust the context.
2. **ABSENCE**: If the <context> does not explicitly or implicitly support an answer, respond exactly: "The answer is not present in the provided text."
3. **CERTAINTY LEVELS**:
   - If the answer is stated verbatim → respond with "Direct: [exact quote/paraphrase]".
   - If the answer requires combining clues → respond with "Inference: [your conclusion]" and briefly explain the chain of reasoning.
4. **CONTRADICTIONS**: If the <context> contains conflicting information, present both sides and state that the text is contradictory. Do not pick a side.
5. **THEMES/TAKEAWAYS**: Only extract a theme if it appears in at least 3 distinct places in the context. Otherwise, refuse.

### OUTPUT FORMAT (strict order)
[ANSWER]: (Your concise, factual answer)
[EVIDENCE]: (The exact quote or close paraphrase from the context)
[CONFIDENCE]: (High / Medium / Low - Low if you had to rely heavily on inference)

---

<context>
${evidenceText}
</context>

Question: ${question}

Answer strictly following the OUTPUT FORMAT and CRITICAL RULES:`;

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

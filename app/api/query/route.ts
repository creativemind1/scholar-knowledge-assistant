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
          heading: c.heading,
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
          .map((h, index) => `Chunk ${index + 1}: ${h.heading ? h.heading : ''} \n${h.evidence}`)
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
The evidence chunks are from the biography of Maulana Ashraf Ali Thanvi (RA) (also called hadrat wala in the text).

**CRITICAL RULES:**

1. **Grounding:** Your answer MUST come strictly from the provided evidence chunks. Do NOT use outside knowledge.

2. **Evidence Selection - MOST IMPORTANT:**
   - FIRST: Scan ALL chunks for the EXACT subject/topic the user is asking about.
   - SECOND: Identify which chunk(s) contain that specific subject.
   - THIRD: IGNORE chunks that don't contain the subject, even if they are longer, more detailed, or more dramatic.
   - The correct chunk is the one that contains the SPECIFIC words or ideas from the question.

3. **Synthesis:**
   - Read ALL chunks carefully.
   - The answer may be in one chunk or scattered across multiple.
   - BUT: If one chunk clearly contains the answer and others don't, use ONLY that one.
   - Do NOT combine information from unrelated chunks just because they are about similar topics

4. **Prioritization:**
   - A short chunk with the exact subject is MORE valuable than a long chunk with a related but different subject.

5. **Natural Language:** Respond as if you're telling someone the answer in person.
   - ✅ *"Moulana named his first child Maryam..."*
   - ❌ *"Based on Chunk 2, Page 102, the evidence states..."*

6. **Context, Not Citation:** You may briefly explain the *reason* behind the answer if the evidence provides it, but do not cite chunk numbers, page numbers, or use phrases like "the relevant text states."

7. **Flow:** If the answer is a simple fact, give a clear, direct sentence. If it's a story, narrate it naturally with a beginning, middle, and end.

8. **Negative:** ONLY say "I couldn't find that information in the provided documents." if NONE of the chunks contain the answer, even after reading all of them carefully.

9. **Interpretation:** The user's question may use different wording than the evidence. When answering, identify equivalent meanings and paraphrases.
   - Examples: journey = travel, purchase = buy, scholar = teacher
   - Do NOT require exact words from the question to appear in the evidence.

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
        console.log(JSON.stringify({
          question,
          answer: fullAnswer,
          contexts: JSON.stringify(evidenceText),
          ground_truth: ""
        }))

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

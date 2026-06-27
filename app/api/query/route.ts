import { NextRequest } from "next/server";
import { getEmbedding, generateAnswer, expandQuery } from "@/lib/ollama";
import { rerankDocuments, verifyAnswer } from "@/lib/ollamaHelpers";
import {
  expandChunks,
  findRelevantChunks,
  getDB,
  hasData,
  isMetadataChunk,
} from "@/lib/vectorStore";
import { ashrafAliThanviPrompt } from "@/lib/prompts/prompts";



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

        // 1. Embed original question (for cosine similarity)
        const questionEmbedding = await getEmbedding(question, "query");

        // 2. Expand query (for BM25 keyword matching)
        const expandedQuestion = await expandQuery(question);

        // 3. Retrieve using BOTH original + expanded
        const [originalResults, expandedResults] = await Promise.all([
          findRelevantChunks(questionEmbedding, question),
          findRelevantChunks(questionEmbedding, expandedQuestion)
        ]);

        // 4. Merge, deduplicate and sort by best score
        const seen = new Set();
        const mergedResults = [...originalResults, ...expandedResults]
          .filter(r => {
            if (seen.has(r.chunk.id)) return false;
            seen.add(r.chunk.id);
            return true;
          })
          .sort((a, b) => (b.combined ?? b.cosine) - (a.combined ?? a.cosine));

        const bestScore = mergedResults[0]?.combined ?? mergedResults[0]?.cosine ?? 0;

        if (bestScore < 0.25 || mergedResults.length === 0) {
          await send({
            type: "answer",
            text: "The biography does not contain information about this.",
          });
          await send({ type: "done" });
          return;
        }

        // 5. Filter metadata chunks
        const relevantChunks = mergedResults
          .map((r) => r.chunk)
          .filter((c) => !isMetadataChunk(c.text));

        // 6. Rerank top 20 candidates
        // const candidates = relevantChunks.slice(0, 20);
        //const rerankedChunks = await rerankDocuments(question, relevantChunks);

        // console.log("Merged results:", relevantChunks.map(item => {
        //   return {
        //     id: item.id,
        //     page: item.page
        //   }
        // }));

        // return;

        // // 7. Take top 8 after reranking
        const retrieved = relevantChunks.slice(0, 3);

        // const topChunks = expandChunks(
        //   rerankedChunks,
        //   db.chunks,
        //   5,
        //   20
        // );



        const hits = retrieved.map((c) => ({
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

        // 8. Send sources preview
        const sources = hits.map((c) => ({
          text: c.evidence.slice(0, 200) + (c.evidence.length > 200 ? "..." : ""),
          page: c.page,
        }));
        await send({ type: "sources", sources });

        // 9. Build evidence text for prompt
        // const evidenceText = hits
        //   .map((h, index) =>
        //     `Chunk ${index + 1}: (Page#: ${h.page - 2}) ${h.heading ? h.heading : ''} \n${h.evidence}`
        //   )
        //   .join("\n\n---\n\n");



        const evidenceText = hits.map((chunk, index) => `
[EVIDENCE ${index + 1}]
Page: ${chunk.page - 2}
Text:
${chunk.evidence}
`).join("\n\n---\n\n");

        console.log(evidenceText, "=======evidenceText=======");

        // 10. Generate answer using original question
        const start = Date.now();
        const prompt = ashrafAliThanviPrompt(question, evidenceText);

        let fullAnswer = "";
        await generateAnswer(prompt, async (chunk) => {
          fullAnswer += chunk;
          await send({ type: "chunk", text: chunk });
        });

        if (!fullAnswer.trim()) {
          fullAnswer = "The biography does not contain information about this.";
          await send({ type: "chunk", text: fullAnswer });
        }

        // 11. Verify answer against chunks
        const verification = verifyAnswer(fullAnswer, hits);
        await send({
          type: "verification",
          verified: verification.verified,
          ...(verification.missing ? { missing: verification.missing } : {}),
        });

        await send({ type: "done" });

        // 12. Log for RAGAS evaluation
        console.log(JSON.stringify({
          question,
          answer: fullAnswer,
          contexts: hits.map(h => h.evidence), // array of strings for RAGAS
          ground_truth: ""
        }));

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

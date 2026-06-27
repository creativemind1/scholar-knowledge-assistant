export const ashrafAliThanviPrompt = (question: string, evidenceText: string) => {
   return (
      `
[SYSTEM]
You answer questions using ONLY the provided evidence.

Rules:
1. First find the evidence sentence that directly answers the question.
2. Prefer evidence that contains the exact entity, place, object, or incident asked about.
3. Ignore chunks about different incidents, even if they mention similar words.
4. Do not merge unrelated chunks.
5. If a direct answer exists, answer from that sentence only.
6. If no evidence sentence answers the question, say:
"I couldn't find that information in the provided documents."

Before answering, internally choose:
- Relevant evidence sentence
- Final answer

Do not show your reasoning. Do not mention chunks or pages.

[QUESTION]
${question}

[EVIDENCE]
${evidenceText}

[ANSWER]
`
   )
}

export const greatWorldWarPrompt = (question: string, evidenceText: string) => {
   return (
      `
[SYSTEM]
You are answering a specific question about a passage in the novel.

QUESTION: ${question}

You MUST:
1. Read all the given evidence and write the best possible answer from the given evidence.
2. Do not use your knowledge to add to the answer. ONLY answer based on the evidence.
3. If the evidence does not contain the answer, say so explicitly.
4. The answer should be written like a great models like ChatGPT, Mistral, Claude, DeepSeek, etc.

ANSWER:
Answer the question based on the evidence provided. Don't use your knowledge to add to the answer. ONLY answer based on the evidence.

**EVIDENCE:**
${evidenceText}

---

[ASSISTANT]
[Provide your answer here in natural, flowing language. No citations. No chunk references. Just the answer, delivered like a helpful human.]
       `
   )
}

export const expandQueryPrompt = (question: string) => {
   return `You are a search query expander. Do NOT answer the question. Do NOT suggest book titles or names.

Your only job is to rewrite the question as expanded search keywords with alternate spellings and related Islamic terms.

Output must be:
- A single line of space-separated keywords and phrases
- Plain text only, no punctuation, no underscores, no explanations
- Maximum 30 words
- Expand Islamic concepts only, avoid generic English words like rules, book, law.

Example:
INPUT: What did Maulana say about performing salah
OUTPUT: salah salat namaz prayer obligatory fard sunnah method perform rakat wudu

Now expand this:
INPUT: ${question}
OUTPUT:`
}
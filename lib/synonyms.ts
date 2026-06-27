// synonyms.ts

const STOPWORDS = new Set([
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "of",
    "to", "and", "or", "what", "who", "when", "where", "how", "why",
    "did", "does", "do", "this", "that", "with", "for", "as", "it", "be",
]);

async function getWord(word: string): Promise<string[]> {
    const url = `https://www.dictionaryapi.com/api/v3/references/thesaurus/json/${word}?key=${process.env.MW_API_KEY}`;

    try {
        const res = await fetch(url);
        const data = await res.json(); // <-- this was missing
        const firstSenseSynonyms: string[] = data[0]?.meta?.syns?.[0] || [];

        // Cap to top 3 to keep expansion tight
        return firstSenseSynonyms.slice(0, 3);
    } catch (err) {
        console.error(`Synonym lookup failed for "${word}":`, err);
        return [];
    }
}

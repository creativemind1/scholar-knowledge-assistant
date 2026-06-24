// lib/build-chunks-from-items.js

interface LlamaParseItem {
    type: "heading" | "text" | "footer" | "list" | "table" | "image";
    md: string;
    value: string;
    level?: number;
    bbox?: any[];
    items: LlamaParseItem[];
}

interface LlamaParsePage {
    page_number: number;
    items: LlamaParseItem[];
    page_width: number;
    page_height: number;
    success: boolean;
}

interface RAGChunk {
    id: string;
    page: number;
    heading: string;
    headingLevel: number;
    text: string;
    embedText: string;
    metadata: {
        sectionTitle: string;
        itemCount: number;
        wordCount: number;
        entities: string[];
    };
}

/**
 * Build RAG chunks from LlamaParse's items-based format
 * Groups items by headings to create topic-based chunks
 */
export function buildRAGChunksFromItems(pages: LlamaParsePage[]): RAGChunk[] {
    const chunks: RAGChunk[] = [];
    let chunkCounter = 0;

    for (const page of pages) {
        // Skip unwanted pages
        if (page.page_number < 4 || page.page_number === 7 || page.page_number === 8) continue;

        // ⭐ Process items on this page
        const pageChunks = processPageItems(page);

        // ⭐ Add to overall chunks
        for (const chunk of pageChunks) {
            const cleanText = chunk.text.trim();
            if (!cleanText) continue;

            chunks.push({
                id: `chunk_${chunkCounter++}`,
                page: page.page_number,
                heading: chunk.heading || 'Untitled Section',
                headingLevel: chunk.headingLevel || 0,
                text: cleanText,
                embedText: cleanText,
                metadata: {
                    sectionTitle: chunk.heading || 'Untitled Section',
                    itemCount: chunk.itemCount || 0,
                    wordCount: cleanText.split(/\s+/).length,
                    entities: extractEntities(cleanText)
                }
            });
        }
    }

    console.log(`✅ Created ${chunks.length} chunks from ${pages.length} pages`);
    return chunks;
}

/**
 * Process items on a single page and group by headings
 */
function processPageItems(page: LlamaParsePage): {
    heading: string;
    headingLevel: number;
    text: string;
    itemCount: number;
}[] {
    const sections: {
        heading: string;
        headingLevel: number;
        text: string;
        itemCount: number;
    }[] = [];

    let currentHeading = '';
    let currentLevel = 0;
    let currentText = '';
    let currentItemCount = 0;
    let hasContent = false;

    // Helper function to remove HTML tags
    function removeHtml(text: string): string {
        if (!text) return '';
        return text.replace(/<[^>]*>/g, '');
    }

    // Helper function to normalize diacritics
    function normalizeDiacritics(text: string): string {
        if (!text) return '';

        // NFKD normalization decomposes characters (e.g., ā → a + ̄)
        // Then remove the combining diacritical marks
        let normalized = text.normalize('NFKD').replace(/[\u0300-\u036f]/g, '');

        // Additional manual mapping for common Arabic transliteration characters
        const diacriticMap: { [key: string]: string } = {
            'ā': 'a',
            'ī': 'i',
            'ū': 'u',
            'ḥ': 'h',
            'ṭ': 't',
            'ṣ': 's',
            'ḍ': 'd',
            'ẓ': 'z',
            'ʿ': "'",
            'ʾ': "'",
            '‘': "'",
            '’': "'",
            '“': '"',
            '”': '"',
        };

        normalized = normalized.replace(/[āīūḥṭṣḍẓʿʾ‘’“”]/g, (match) => {
            return diacriticMap[match] || match;
        });

        return normalized;
    }

    // Helper function to clean text (HTML + diacritics)
    function cleanText(text: string): string {
        if (!text) return '';
        return normalizeDiacritics(removeHtml(text));
    }

    for (const item of page.items) {
        // Skip footer items
        if (item.type === 'footer') continue;

        // If we find a heading, start a new section
        if (item.type === 'heading') {
            // Save previous section if it has content
            if (hasContent && currentText.trim()) {
                sections.push({
                    heading: cleanText(currentHeading) || 'Untitled Section',
                    headingLevel: currentLevel,
                    text: cleanText(currentText.trim()),
                    itemCount: currentItemCount
                });
            }

            // Start new section with this heading
            currentHeading = item.value || item.md || '';
            currentLevel = item.level || 0;
            currentText = '';
            currentItemCount = 0;
            hasContent = false;

            // Debug: Log heading found
            if (currentHeading.includes('I’TIKAAF') || currentHeading.includes('SPRING') || currentHeading.includes('HAZRAT')) {
                console.log(`🔍 Found heading on page ${page.page_number}: "${cleanText(currentHeading)}"`);
            }
        }
        // ⭐ Everything else (text, list, list_item, etc.) gets added to current section
        else {
            let textContent = '';

            // Handle list items
            if (item.type === 'list') {
                // Get all list items as a string
                if (item.items && Array.isArray(item.items)) {
                    textContent = item.items.map(i => cleanText(i.value || i.md || '')).join(' ');
                } else {
                    textContent = cleanText(item.value || item.md || '');
                }
            }

            // Regular text
            else {
                textContent = cleanText(item.value || item.md || '');
            }

            if (textContent.trim()) {
                currentText += (currentText ? ' ' : '') + textContent;
                currentItemCount++;
                hasContent = true;
            }
        }
    }

    // Don't forget the last section
    if (hasContent && currentText.trim()) {
        sections.push({
            heading: cleanText(currentHeading) || 'Untitled Section',
            headingLevel: currentLevel,
            text: cleanText(currentText.trim()),
            itemCount: currentItemCount
        });
    }

    // If no sections were created, create one chunk for the whole page
    if (sections.length === 0) {
        const allText = page.items
            .filter(item => item.type !== 'footer')
            .map(item => {
                if (item.type === 'list' && item.items) {
                    return item.items.map(i => cleanText(i.value || i.md || '')).join(' ');
                }
                return cleanText(item.value || item.md || '');
            })
            .join(' ')
            .trim();

        if (allText) {
            sections.push({
                heading: 'Untitled Section',
                headingLevel: 0,
                text: cleanText(allText),
                itemCount: page.items.length
            });
        }
    }

    return sections;
}

/**
 * Extract entities (people, places, things) from text
 */
function extractEntities(text: string): string[] {
    // Look for capitalized words that might be names or places
    const entities = text.match(/[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/g) || [];
    return [...new Set(entities)].slice(0, 10);
}
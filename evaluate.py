# evaluate.py
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())  # ← fixes the error

from eval_data import samples

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# LLM judge
llm = LangchainLLMWrapper(Ollama(model="llama3.2"))

# Embeddings (local, no OpenAI needed)
embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

data = {
    "question":     [s["question"]     for s in samples],
    "contexts":     [s["contexts"]     for s in samples],
    "answer":       [s["answer"]       for s in samples],
    "ground_truth": [s["ground_truth"] for s in samples],
}

dataset = Dataset.from_dict(data)

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_entity_recall,
        answer_similarity,
        answer_correctness,
    ],
    llm=llm,
    embeddings=embeddings
)

print(result)
result.to_pandas().to_csv("results.csv", index=False)
print("✅ Results saved to results.csv")
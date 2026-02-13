import logging
import os
import requests
from typing import List

from embeddings import LocalEmbedding
from vector_store import FAISSVectorStore
from document_loader import load_documents
from text_cleaner import clean_text
from chunking import chunk_text
from prompts import build_prompt_v2


SIMILARITY_THRESHOLD = 0.30


class RAGPipeline:
    def __init__(self, llm_model: str, temperature: float):
        self.llm_model = llm_model
        self.temperature = temperature

        self.embedder = LocalEmbedding()
        self.vector_store = FAISSVectorStore()

    # ---------------------------------------------------------
    # Build Knowledge Base
    # ---------------------------------------------------------
    def build_knowledge_base(self, folder_path: str):
        logging.info("Loading documents...")

        documents = load_documents(folder_path)
        all_chunks = []

        for filename, content in documents:
            cleaned = clean_text(content)
            chunks = chunk_text(cleaned, filename)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No documents found in data folder.")

        texts = [c["text"] for c in all_chunks]

        embeddings = self.embedder.embed_texts(texts)

        self.vector_store.add_embeddings(embeddings, all_chunks)

        logging.info("Knowledge base built successfully.")

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 3):
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k)
        return results

    # ---------------------------------------------------------
    # Confidence Scoring
    # ---------------------------------------------------------
    def calculate_confidence(self, similarities: List[float]) -> float:
        if not similarities:
            return 0.0

        avg_similarity = sum(similarities) / len(similarities)
        support_factor = min(len(similarities) / 3, 1.0)

        confidence = (avg_similarity * 0.7) + (support_factor * 0.3)

        return round(confidence, 3)

    # ---------------------------------------------------------
    # OpenRouter LLM Call
    # ---------------------------------------------------------
    def call_openrouter(self, prompt: str) -> str:

        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Company Policy RAG Assistant"
        }

        payload = {
            "model": self.llm_model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": "You are a strict policy assistant."},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"OpenRouter Error ({response.status_code}): {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"]

    # ---------------------------------------------------------
    # Main QA Method
    # ---------------------------------------------------------
    def answer_question(self, question: str) -> str:

        retrieved = self.retrieve(question)
        print("\nRetrieved results:")
        for score, metadata in retrieved:
            print("Score:", score, "| File:", metadata["filename"])

        if not retrieved:
            return "No relevant information found in the knowledge base."

        filtered_chunks = []
        similarities = []

        for score, metadata in retrieved:
            if score >= SIMILARITY_THRESHOLD:
                filtered_chunks.append((score, metadata))
                similarities.append(score)

        if not filtered_chunks:
            return "No relevant information found in the knowledge base."

        context = ""
        retrieval_summary = "\n\n## Retrieved Evidence\n"

        for score, metadata in filtered_chunks:
            preview = metadata["text"][:200].replace("\n", " ")

            retrieval_summary += (
                f"- {metadata['filename']} | "
                f"Chunk {metadata['chunk_id']} | "
                f"Similarity: {score:.3f}\n"
                f"  Preview: {preview}...\n"
            )

            context += f"""
Source: {metadata['filename']}
Chunk ID: {metadata['chunk_id']}

{metadata['text']}

"""

        confidence = self.calculate_confidence(similarities)

        prompt = build_prompt_v2(question, context)

        try:
            final_answer = self.call_openrouter(prompt)
        except Exception as e:
            return f"⚠️ LLM Error:\n{str(e)}"

        return f"""{retrieval_summary}

{final_answer}

---
## Confidence Score
{confidence}
"""
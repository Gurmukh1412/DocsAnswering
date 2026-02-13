from sentence_transformers import SentenceTransformer
from typing import List
import logging


class LocalEmbedding:
    """
    Free local embedding model using SentenceTransformers.
    Model: all-MiniLM-L6-v2
    Small, fast, and excellent for RAG.
    """

    def __init__(self):
        logging.info("Loading local embedding model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        logging.info("Generating embeddings locally...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        embedding = self.model.encode([query])
        return embedding[0].tolist()
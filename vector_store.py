import faiss
import numpy as np
from typing import List, Tuple


class FAISSVectorStore:
    """
    FAISS using cosine similarity (normalized vectors + inner product).
    """

    def __init__(self):
        self.index = None
        self.metadata = []

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms

    def add_embeddings(self, embeddings: List[List[float]], metadata: List[dict]) -> None:
        vectors = np.array(embeddings).astype("float32")
        vectors = self._normalize(vectors)

        dimension = vectors.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatIP(dimension)

        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_embedding: List[float], top_k: int) -> List[Tuple[float, dict]]:
        vector = np.array([query_embedding]).astype("float32")
        vector = self._normalize(vector)

        similarities, indices = self.index.search(vector, top_k)

        results = []
        for score, idx in zip(similarities[0], indices[0]):
            if idx < len(self.metadata):
                results.append((float(score), self.metadata[idx]))

        return results
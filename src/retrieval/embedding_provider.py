"""
This module provides a centralized way to initialize and access
embeddings models used across the retrieval system.

It decouples embeddings logic from retrievers and vector stores,
allowing easy replacement of embedding models.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import RAGConfig


class EmbeddingProvider:
    """
    Handles initialization and reuse of embedding models.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._embeddings = None

    def get_embeddings(self):
        """
        Lazily initialize and return embeddings.
        """
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": self.config.device},
                encode_kwargs={"normalize_embeddings": False},
            )
        return self._embeddings
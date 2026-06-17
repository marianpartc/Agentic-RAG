"""
This module implements a simple vector-based retriever.

It uses embeddings and a vector store (FAISS) to perform
similarity serch over documents.

This serves as the baseline (naive) retrieval strategy,
witch can later be extended or replace by more advanced
approaches.
"""

from typing import List
from langchain_core.documents import Document

from src.config import RAGConfig
from .base_retriever import BaseRetriever
from .embedding_provider import EmbeddingProvider
from .vector_store import VectorStore


class NaiveRetriever(BaseRetriever):
    """
    Basic vector similarity retriever using FAISS.
    """

    def __init__(self, config: RAGConfig, documents: List[Document]):
        self.config = config

        # Initialize embeddings
        self.embedding_provider = EmbeddingProvider(config)
        embeddings = self.embedding_provider.get_embeddings()

        # Build vector store
        self.vector_store = VectorStore(embeddings)
        self.vector_store.build(documents)

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve top-k similar documents for the query.
        """
        return self.vector_store.similarity_search(
            query,
            k=self.config.num_retrieved_docs
        )
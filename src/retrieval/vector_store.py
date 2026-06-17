"""
This module defines a wrapper around vector databases.

It abstracts vector storage and similarity serch operations, 
allowing different backends (FAISS, Chroma, etc) to be used 
without changing retriever logic.
"""

from typing import List
from langchain.vectorstores import FAISS
from langchain_core.documents import Document


class VectorStore:
    """
    Wrapper for vector database operations.
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.db = None

    def build(self, documents: List[Document]):
        """
        Build the vector store from documents.
        """
        self.db = FAISS.from_documents(documents, self.embeddings)

    def similarity_search(self, query: str, k: int) -> List[Document]:
        """
        Perform similarity search in the vector store.
        """
        return self.db.similarity_search(query, k=k)
"""
This module define the abstract interface for all retrieval strategies.

It ensure that different retrievers (vector-based, graph-based) share
a common contract, making them interchangable within the RAG pipeline.
"""


from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.
    """

    @abstractmethod
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a given query.
        """
        pass
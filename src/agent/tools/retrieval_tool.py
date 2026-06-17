"""
This module wraps the retriever into a tool usable by the agent.

It transforms retrieved documents into a structured output
that includes context and metadata for traceability.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from src.retrieval.base_retriever import BaseRetriever


class RetrievalTool:
    """
    Tool to retrieve relevant documents.
    """

    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute retrieval and return structured results.
        """
        docs: List[Document] = self.retriever.retrieve(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        sources = [doc.metadata for doc in docs]

        return {
            "query": query,
            "context": context,
            "sources": sources,
            "documents": docs
        }
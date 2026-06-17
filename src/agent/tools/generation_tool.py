"""
This module wraps the generator into a tool usable by the agent.

It takes the query and retrieved documents and produces
a final grounded answer.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from src.generation.base_generator import BaseGenerator


class GenerationTool:
    """
    Tool to generate final answers using the LLM.
    """

    def __init__(self, generator: BaseGenerator):
        self.generator = generator

    def run(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """
        Generate answer from query and documents.
        """
        documents = documents if documents is not None else []
        answer = self.generator.generate(query, documents)

        return {
            "query": query,
            "answer": answer,
            "documents": documents
        }
"""
This module defines the abstract interface for all generators.

A generator is responsible for producing a final answer given: 
- A query
- Retrieved context

This abstraction allows different LLM providers (OpenAI, local models)
to be used interchangeably.
"""

from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class BaseGenerator(ABC):
    """
    Abstract base class for all generators.
    """

    @abstractmethod
    def generate(self, query: str, context_docs: List[Document]) -> str:
        """
        Generate a response given a query and retrieved documents.
        """
        pass

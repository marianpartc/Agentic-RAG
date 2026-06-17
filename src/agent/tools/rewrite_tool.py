"""
This module defines a tool to rewrite or improve user queries
uaing the same LLM as the generator.
"""

from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from src.generation.base_generator import BaseGenerator


class RewriteTool:
    """
    Tool to rewrite queries for better retrieval.
    """

    def __init__(self, generator: BaseGenerator):
        self.llm = generator.llm

    def run(self, query: str) -> Dict[str, Any]:
        """
        Rewrite the user query to improve clarity and retrieval quality.
        """

        template = """
        Rewrite the following query to make it clearer, more specific,
        and optimized for document retrieval.

        Original query:
        {query}

        Rewritten query:
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["query"]
        )

        chain = prompt | self.llm

        response = chain.invoke({"query": query})

        rewritten_query = response.content.strip()

        return {
            "original_query": query,
            "rewritten_query": rewritten_query
        }
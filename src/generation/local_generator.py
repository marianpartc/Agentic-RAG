"""
This module implements a generator using a local LLM (via OpenAI-compatible API).

It connects to a locally hosted model (LM Studio).
"""

from typing import List
from openai import OpenAI
from langchain_core.documents import Document

from src.config import RAGConfig
from .base_generator import BaseGenerator


class LocalGenerator(BaseGenerator):
    """
    Generator implementation using a local LLM.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = self._init_local_llm()

    def _init_local_llm(self):
        """
        Initialize local LLM client.
        """
        return OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio"
        )

    def generate(self, query: str, context_docs: List[Document]) -> str:
        """
        Generate response using local LLM.
        """

        context = "\n\n".join(doc.page_content for doc in context_docs)

        template = """You are a helpful academic assistant. Your name is ESIA.

INSTRUCTIONS:
1. If the user greets you or asks a general "small talk" question, respond naturally and politely.
2. For any factual or academic question, use the provided context as your primary source.
3. If you use information from the context, you MUST include a final citation at the end of your answer in this format:
   (Page X) or (Pages X, Y, Z)
4. If the question cannot be answered using the context, inform the user honestly but offer general knowledge if appropriate.

CITATION RULES:
- Always place citations at the END of the answer.
- If multiple pages are used, group them: (Pages 5, 6, 8)
- Do NOT mention DOC numbers.
- Only cite pages that were actually used.

ADVANCED CITATION MODE:
- If the user explicitly asks for references, citations, or bibliography:
    - Provide full references using available metadata (author, year, title, DOI if available).
    - Use a clean academic format (APA-style preferred unless specified otherwise).
    - You may include page numbers if relevant.

Context:
{context}

Chat history:
{chat_history}

Question:
{question}

Answer:"""

        response = self.llm.chat.completions.create(
            model='unsloth/deepseek-r1-distill-qwen-7b',
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": template.format(context=context, question=query)
                }
            ],
            temperature=self.config.temperature
        )

        return response.choices[0].message.content

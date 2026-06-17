"""
This module implements a generator using OpenAI GPT models. 

It uses LangChain chat models and maintains conversation memory.
"""


import os
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory

from src.config import RAGConfig
from .base_generator import BaseGenerator


class OpenAIGenerator(BaseGenerator):
    """
    Generator implementation using OpenAI GPT models.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = self._init_gpt()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def _init_gpt(self):
        """
        Initialize GPT model.
        """
        return init_chat_model(
            model="gpt-4.1-nano",
            model_provider="openai",
            openai_api_key=os.environ["OPENAI_API_KEY"],
            temperature=self.config.temperature
        )
    
    def _build_context(self, context_docs: List[Document]) -> str:
        """
        Build structured context including metadata for citation.
        """

        context_blocks = []

        for i, doc in enumerate(context_docs):
            metadata = doc.metadata

            page = metadata.get("page_number", "?")
            author = metadata.get("author_real", "Unknown")
            year = metadata.get("year", "Unknown")
            title = metadata.get("title", "Unknown")

            block = f"""
[DOC {i+1} | Page {page}]
Author: {author}
Year: {year}
Title: {title}

{doc.page_content}
"""
            context_blocks.append(block)

        return "\n\n".join(context_blocks)


    def generate(self, query: str, context_docs: List[Document]) -> str:
        """
        Generate response using GPT with context, citation and memory.
        """

        # Use structured context with metadata
        context = self._build_context(context_docs)

        chat_history = self.memory.load_memory_variables({})["chat_history"]

        template = """
You are a helpful academic assistant. Your name is ESIA.

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

Answer:
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )

        chain = prompt | self.llm

        response = chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "question": query
        })

        self.memory.save_context(
            {"input": query},
            {"output": response.content}
        )

        return response.content
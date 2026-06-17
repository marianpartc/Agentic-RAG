"""
This module handles loading PDF documents into LangChain Document objects.
It abstracts away the underlying PDF loading implementation.
"""

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from typing import List

class PDFLoader:

    def load(self, file_path: str) -> List[Document]:
        loader = PyMuPDFLoader(file_path)
        return loader.load()

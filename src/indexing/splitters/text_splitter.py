"""
This module is responsible for splitting documents into smaller chunks
using configurable chunk size and overlap.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class TextSplitter:

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap 
        )    
    
    def split(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)

"""
This module defines the document processing pipeline.

It orchestrates the full workflow:
1. Load raw documents from a source
2. Extract structured metadata (including abstract)
3. Generate semantic tags
4. Enrich documents (metadata + text injection + section detection)
5. Split documents into chunks
6. Generate a verification TXT file (optional debugging artifact)
"""

from typing import List
import os

from langchain_core.documents import Document

from src.config import RAGConfig

from src.indexing.loaders.pdf_loader import PDFLoader
from src.indexing.splitters.text_splitter import TextSplitter
from src.indexing.processors.metadata_extractor import MetadataExtractor
from src.indexing.processors.text_enricher import TextEnricher
from src.indexing.processors.tagger import Tagger


class DocumentProcessor:
    """
    Orchestrates the document processing workflow.
    """

    def __init__(self, config: RAGConfig):
        self.config = config

        # Initialize modular components
        self.loader = PDFLoader()
        self.splitter = TextSplitter(
            config.chunk_size,
            config.chunk_overlap
        )
        self.metadata_extractor = MetadataExtractor()
        self.enricher = TextEnricher()
        self.tagger = Tagger()

    def process(self, file_path: str) -> List[Document]:
        """
        Execute the full document processing pipeline.
        """

        # Load documents (one per page)
        documents = self.loader.load(file_path)

        if not documents:
            return []

        # Extract structured metadata (includes abstract)
        metadata = self.metadata_extractor.extract(file_path)

        # Generate semantic tags (use abstract now)
        tags = self.tagger.generate(
            metadata.get("title"),
            metadata.get("abstract")
        )

        # Enrich documents (metadata + section + text injection)
        if self.config.use_metadata_enrichment:
            documents = self.enricher.enrich(
                documents,
                metadata,
                file_path,
                tags
            )

        # Split documents into chunks
        chunked_docs = self.splitter.split(documents)

        # Generate verification TXT (same behavior as original code)
        if self.config.save_verification_file:
            self._write_verification_file(file_path, chunked_docs)

        return chunked_docs


    def _write_verification_file(self, file_path: str, chunked_docs: List[Document]) -> None:
        """
        Writes a TXT file containing chunk text and metadata for debugging.
        """
        dir_name = os.path.dirname(file_path) or "."
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_txt_path = os.path.join(dir_name, f"{base_name}_VERIFICACION.txt")

        with open(output_txt_path, "w", encoding="utf-8") as f:
            for idx, doc in enumerate(chunked_docs):
                f.write(f"================= CHUNK {idx} =================\n")
                f.write(">>> TEXT:\n")
                f.write(doc.page_content)
                f.write("\n\n")
                f.write(">>> METADATA:\n")
                for key, value in doc.metadata.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n\n-----------------------------------------\n\n")

        print(f"Verification file created: {output_txt_path}\n")
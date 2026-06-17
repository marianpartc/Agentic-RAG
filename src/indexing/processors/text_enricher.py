"""
This module enriches document text by injecting structured metadata
into the content before embedding. This can improve retrieval quality
by making metadata semantically searchable.

It enriches LangChain Document objects by:
1. Injecting metadata into the text content (for better embeddings)
2. Updating structured metadata fields at the document level
3. Detecting semantic sections within the text
"""

from typing import List, Dict
from langchain_core.documents import Document


class TextEnricher:

    def enrich(self, documents: List[Document], metadata: Dict, file_path: str, tags: List[str]) -> List[Document]:
        enriched_docs = []

        for i, doc in enumerate(documents):

            # --- Update structured metadata ---
            doc.metadata.update(metadata)
            doc.metadata["tags"] = tags
            doc.metadata["page_number"] = doc.metadata.get("page", i) + 1
            doc.metadata["doc_id"] = file_path

            section = self._detect_section(doc.page_content)
            doc.metadata["section"] = section

            # --- Inject metadata into text (only first chunk) ---
            metadata_text = []

            if metadata.get("title"):
                metadata_text.append(f"Title: {metadata['title']}")

            if metadata.get("author_real"):
                metadata_text.append(f"Author: {metadata['author_real']}")

            if metadata.get("year"):
                metadata_text.append(f"Year: {metadata['year']}")

            if metadata.get("doi"):
                metadata_text.append(f"DOI: {metadata['doi']}")

            if metadata_text and i == 0:
                metadata_block = "\n".join(metadata_text) + "\n\n"
                doc.page_content = metadata_block + doc.page_content

            enriched_docs.append(doc)

        return enriched_docs

    def _detect_section(self, text: str) -> str:
        """
        Detects the semantic section of a document based on keyword matching.
        """
        lower = text.lower()

        sections = {
            "resumen": "Resumen",
            "abstract": "Resumen",
            "summary": "Resumen",
            "introducción": "Introducción",
            "introduction": "Introducción",
            "metodología": "Metodología",
            "methods": "Metodología",
            "resultados": "Resultados",
            "results": "Resultados",
            "discusión": "Discusión",
            "discussion": "Discusión",
            "conclusiones": "Conclusiones",
            "conclusion": "Conclusiones",
            "palabras clave": "Palabras clave",
            "keywords": "Palabras clave"
        }

        for key, sec in sections.items():
            if key in lower:
                return sec

        return "General text"

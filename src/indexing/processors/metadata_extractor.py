"""
This module extracts structured metadata from PDF documents,
such as title, authors, year, DOI, emails, ORCID identifiers 
and abstract.It also includes heuristics to infer missing metadata.
"""

import re
from typing import Dict, Optional, List
from pypdf import PdfReader

DOI_REGEX = r"10\.\d{4,9}\/[-._;()\/:A-Za-z0-9]+"
EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
ORCID_REGEX = r"https?:\/\/orcid\.org\/[\d\-]{15,}"
ISSN_REGEX = r"\d{4}-\d{3}[\dX]"

class MetadataExtractor:

    def extract(self, file_path: str) -> Dict:
        reader = PdfReader(file_path)
        meta = reader.metadata or {}

        first_page = reader.pages[0].extract_text() or ""

        doi_match = re.search(DOI_REGEX, first_page)
        doi = doi_match.group(0) if doi_match else None

        emails = re.findall(EMAIL_REGEX, first_page)

        orcids = list(set(re.findall(ORCID_REGEX, first_page)))

        year_match = re.search(r"(19|20)\d{2}", first_page)
        year = int(year_match.group(0)) if year_match else None

        author_real = self._guess_authors_from_first_page(first_page)

        title = meta.get("/Title") or "Unknown title"

        # Abstract from first pages
        full_text = "\n\n".join(
            (reader.pages[i].extract_text() or "") for i in range(min(3, len(reader.pages)))
        )
        abstract = self._extract_abstract(full_text)

        return {
            "source": file_path,
            "title": title,
            "author_real": author_real or "Unknown author",
            "year": year,
            "doi": doi,
            "emails": emails,
            "orcids": orcids,
            "issn": meta.get("/ISSN"),
            "abstract": abstract
        }

    def _guess_authors_from_first_page(self, text: str) -> Optional[str]:
        """
        Heuristic to detect author names from the first page.
        It looks for 2–4 word capitalized name patterns and filters noisy lines.
        """
        if not text:
            return None

        lines = [l.strip() for l in text.splitlines() if l.strip()]

        bad_keywords = [
            "universidad", "facultad", "coordinación", "división",
            "cd. mx", "méxico", "unam", "revista", "issn",
            "departamento", "dirección"
        ]

        name_pattern = re.compile(
            r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?: [A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3}"
        )

        candidates: List[str] = []

        for line in lines:
            lower = line.lower()

            if len(line) > 80:
                continue

            if any(bad in lower for bad in bad_keywords):
                continue

            matches = name_pattern.findall(line)
            if matches:
                candidates.extend(matches)

        if candidates:
            unique = list(dict.fromkeys(candidates))
            return ", ".join(unique)

        return None

    def _extract_abstract(self, text: str) -> Optional[str]:
        """
        Extract abstract by locating common headers such as 'abstract', 'resumen', or 'summary'.
        """
        if not text:
            return None

        lower = text.lower()
        headers = ["resumen", "abstract", "summary"]
        end_markers = ["palabras clave", "keywords"]

        for header in headers:
            idx = lower.find(header)
            if idx == -1:
                continue

            start = idx + len(header)

            ends = []
            for marker in end_markers:
                j = lower.find(marker, start)
                if j != -1:
                    ends.append(j)

            end = min(ends) if ends else min(len(text), start + 2000)

            abstract_text = text[start:end].strip()
            if abstract_text:
                return abstract_text

        return None
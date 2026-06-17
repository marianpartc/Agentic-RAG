"""
This module generates semantic tags from document title and abstract.
"""

from typing import List, Optional

class Tagger:

    def generate(self, title: Optional[str], abstract: Optional[str]) -> List[str]:
        text = ((title or "") + " " + (abstract or "")).lower()
        tags: List[str] = []

        if "inteligencia artificial" in text or "artificial intelligence" in text:
            tags.append("ai")

        if any(word in text for word in [
            "aprendizaje automático", "machine learning", "aprendizaje de máquina"
        ]):
            tags.append("machine-learning")

        if any(word in text for word in [
            "aprendizaje profundo", "deep learning"
        ]):
            tags.append("deep-learning")

        if any(word in text for word in [
            "procesamiento del lenguaje natural", "pln", "nlp", "natural language processing"
        ]):
            tags.append("nlp")

        if any(word in text for word in [
            "retrieval augmented generation", "rag"
        ]):
            tags.append("rag")

        if any(word in text for word in [
            "revisión de la literatura", "literature review", "systematic review"
        ]):
            tags.append("literature-review")

        if not tags and (title or abstract):
            tags.append("general-paper")

        return tags

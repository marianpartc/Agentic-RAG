"""
This module provides a factory to select the appropriate generator
based on configuration.
"""

from src.config import RAGConfig
from .openai_generator import OpenAIGenerator
from .local_generator import LocalGenerator


class GeneratorFactory:
    """
    Factory to create generator instances.
    """

    @staticmethod
    def get_generator(config: RAGConfig):
        """
        Return the appropriate generator based on config.
        """

        if config.generator_type == "openai":
            return OpenAIGenerator(config)

        elif config.generator_type == "local":
            return LocalGenerator(config)

        else:
            raise ValueError(f"Unknown generator type: {config.generator_type}")
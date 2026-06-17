"""
This module defines the configuration for the RAG system.

It centralizes all parameters related to:
- Document processing
- Embeddings
- Retrieval 
- Generation
- Pipeline behavior
"""

class RAGConfig:
    """Configuration class for RAG system."""

    def __init__(self):

        #### Document processing ####
        self.chunk_size = 3000
        self.chunk_overlap = 200
        # Enable/disable metadata injection into text
        self.use_metadata_enrichment = True

        #### Embeddings ####
        #self.embedding_model = "./data/embeddings/all-mpnet-base-v2"
        self.embedding_model = "sentence-transformers/all-mpnet-base-v2"
        self.device = 'cpu'

        #### Retrieval ####
        self.vector_store_type = "faiss"  # "chroma", "weaviate"
        self.num_retrieved_docs = 1
        self.retriever_type = "naive"  # "graph", "agentic"

        #### Generation ####
        self.temperature = 0.7
        self.generator_type = "openai"  # or "local"

        #### Debug/Logging ####
        self.save_verification_file = False


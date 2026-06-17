"""
Main entry point for the modular RAG system.

This script:
    1. Processes documents / Indexing
    2. Initializes retriever
    3. Initializes generator (OpenAI or local)
    4. Allows selecting between:
        - Classic RAG (retrieve -> generate)
        - Agent-based RAG (reasoning with tools)
"""

import sys
from src.config import RAGConfig
from src.indexing.document_processor import DocumentProcessor
from src.retrieval.naive_retriever import NaiveRetriever
from src.generation.generator_factory import GeneratorFactory

# Agent imports
from src.agent.orchestrator import AgentOrchestrator
from src.agent.tools.retrieval_tool import RetrievalTool
from src.agent.tools.generation_tool import GenerationTool
from src.agent.tools.rewrite_tool import RewriteTool

def main():
    try:
        # Initialize configuration
        config = RAGConfig()

        # Select execution mode
        mode = input("Select mode (classic/agent): ").strip().lower()

        if mode not in ["classic", "agent"]:
            print("Invalid mode. Defaulting to 'classic'.")
            mode = "classic"

        # Process documents
        print("Processing documents...")
        pipeline = DocumentProcessor(config)

        documents = pipeline.process(
            "./data/raw/Article_1.pdf"
        )

        print(f"Loaded {len(documents)} chunks.")

        # Initialize retriever
        print("Initializing retriever...")
        retriever = NaiveRetriever(config, documents)
        
        # Initialize generator
        print(f"Initializing generator ({config.generator_type})...")
        generator = GeneratorFactory.get_generator(config)

        # Initialize agent (only if needed)
        if mode == "agent":
            print("Initializing agent...")

            retrieval_tool = RetrievalTool(retriever)
            generation_tool = GenerationTool(generator)
            rewrite_tool = RewriteTool(generator)

            agent = AgentOrchestrator(
                config,
                retrieval_tool,
                generation_tool,
                rewrite_tool,
                generator.llm
            )

        print("\n" + "=" * 50)
        print(f"RAG system ready in '{mode}' mode.")
        print("Write your query or 'quit' to exit.")
        print("=" * 50 + "\n")

        # Query loop
        while True:
            query = input("\nUser: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "salir"]:
                print("\nAssistant: Bye!")
                break

            # Classic RAG pipeline
            if mode == "classic":
                print("\nRetrieving context...")
                context_docs = retriever.retrieve(query)

                print("Generating response...")
                response = generator.generate(query, context_docs)

            # Agent-based RAG pipeline
            elif mode == "agent":
                print("\nRunning agent...")
                response = agent.run(query)

            print(f"\nAssistant: {response}")

    except Exception as e:
        print(f"\nCritical error: {str(e)}")


if __name__ == "__main__":
    main()
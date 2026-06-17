# Agentic RAG

## Overview

This project explores the evolution of traditional Retrieval-Augmented Generation (RAG) systems toward more autonomous and flexible agentic architectures.

Starting from a modular implementation of a Naive RAG pipeline, individual components are transformed into callable tools that can be orchestrated dynamically by an LLM-based agent. Additionally, query rewriting mechanism ia incorporated to improve retrieval quality.

---

## Design Principles

The project was designed following modular software engineering principles.

Each component can be:

* Used independently
* Replaced by alternative implementations
* Exposed as an agent tool
* Evaluated separately
* Reused in different RAG architectures

---

## Features

*  Modular implementation of a Naive RAG pipeline
*  Independent Retriever module
*  Independent Generator module
*  Query Rewriting module
*  Agent-based orchestration
*  Tool Calling capabilities
*  Iterative Retrieval
*  Configurable number of retrieval iterations

### Future Features

*  Retrieval evaluation module
*  Memory support
*  External tools integration

---

## Evolution of the Architecture

```text
Naive RAG
    │
    ▼
Modular RAG
    │
    ▼
Agentic RAG
```

---

## Agent Workflow & Decision Architecture

The orchestrator agent operates as a dynamic router utilizing a strict **Classification -> Action** reasoning loop. Instead of a rigid linear pipeline, the agent evaluates the user query and the current system state to select the most appropriate tool.

### The Reasoning Loop

1. **Intent Classification:** The agent classifies the user query into one of three distinct categories:
   * **`DOCUMENT-BASED`:** Queries explicitly targeting the indexed knowledge base.
   * **`GENERAL KNOWLEDGE`:** Standard questions outside the document's scope.
   * **`SMALL TALK`:** Greetings and casual interactions.

2. **Dynamic Routing/Decision Rules:**
   * **Retrieval Trigger:** Initiates the `retrieve` tool *only* if the intent is `DOCUMENT-BASED` and no prior context has been gathered.
   * **Query Refinement:** If the user query is ambiguous or unclear, the agent triggers the `rewrite` tool to optimize the search terms before attempting retrieval.
   * **Direct Response Generation:** For `GENERAL KNOWLEDGE` or `SMALL TALK`, it bypasses retrieval and routes directly to the `generate` tool.

3. **Execution Guardrails:**
   * **Strict Tool Dependency:** The agent never answers the user directly; every output must cycle through an explicit action framework (`Thought -> Action -> Action Input`).
   * **Token Efficiency:** To minimize latency and API costs, the agent is restricted from redundant retrieval loops unless strictly necessary.

4. **Final Synthesis:** Once the required context is gathered (or bypassed based on classification), the agent calls the `generate` tool to synthesize and deliver the final answer.

---

## Components

### Retriever

Responsible for obtaining relevant chunks from the knowledge base.

### Generator

Produces the final response based on the retrieved context.

### Query Rewriter

Generates alternative formulations of the user's question to improve retrieval quality.

### Agent

Coordinates the execution of tools and determines the next action to perform.

---

## Author

Marian Partida

Applied Mathematician | AI Engineer

---


### Sources:
- Langchain: [Documentation](https://python.langchain.com/docs/introduction/)
- OpenNotebook: [Github](https://github.com/lfnovo/open-notebook?tab=readme-ov-file)



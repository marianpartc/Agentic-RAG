"""
This module implements the agent orchestrator.

The orchestrator controls the reasoning loop:
- Decides which tool to use
- Executes the tool
- Updates the state
- Repeats until a final answer is produced
"""

from typing import Dict, Any

from langchain_core.prompts import PromptTemplate

from src.config import RAGConfig
from src.agent.prompts.agent_prompt import AGENT_PROMPT
from src.agent.tools.retrieval_tool import RetrievalTool
from src.agent.tools.generation_tool import GenerationTool
from src.agent.tools.rewrite_tool import RewriteTool


class AgentOrchestrator:
    """
    Main agent controller implementing a simple reasoning loop.
    """

    def __init__(
        self,
        config: RAGConfig,
        retrieval_tool: RetrievalTool,
        generation_tool: GenerationTool,
        rewrite_tool: RewriteTool,
        llm
    ):
        self.config = config
        self.retrieval_tool = retrieval_tool
        self.generation_tool = generation_tool
        self.rewrite_tool = rewrite_tool
        self.llm = llm

    def run(self, query: str) -> str:
        """
        Execute the agent reasoning loop with safeguards to avoid infinite loops.
        """

        state: Dict[str, Any] = {
            "query": query,
            "documents": None,
            "context": None,
            "has_document_access": True
        }

        max_steps = 5
        step = 0

        while step < max_steps:
            step += 1

            prompt = PromptTemplate(
                template=AGENT_PROMPT,
                input_variables=["query", "state"]
            )

            chain = prompt | self.llm

            response = chain.invoke({
                "query": state["query"],
                "state": str(state)
            })

            output = response.content

            print(f"\n[Agent reasoning - step {step}]")
            print(output)

            # Parse action
            if "Action:" not in output:
                return "Agent error: No valid action produced."

            action_line = [line for line in output.split("\n") if "Action:" in line][0]
            action = action_line.split("Action:")[-1].strip().lower()

            # SAFEGUARD: If already retrieved, force generate
            if state["documents"] is not None and action == "retrieve":
                action = "generate"

            if action == "rewrite":
                result = self.rewrite_tool.run(state["query"])
                state["query"] = result["rewritten_query"]

            elif action == "retrieve":
                result = self.retrieval_tool.run(state["query"])
                state["documents"] = result["documents"]
                state["context"] = result["context"]

            elif action == "generate":
                result = self.generation_tool.run(
                    state["query"],
                    state["documents"] if state["documents"] is not None else []
                )
                return result["answer"]

            else:
                return "Agent could not determine a valid action."

        # Fallback if max steps reached
        return "Agent stopped due to too many reasoning steps."
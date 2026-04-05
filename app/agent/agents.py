"""Eleven specialized LangChain-based agents for the Paper2Agent pipeline.

Each agent encapsulates a role-specific system prompt, a list of allowed
tools, and a ``run()`` method that delegates to ``langchain_openai.ChatOpenAI``.
"""

import logging
import time
from typing import Any

from config import Config

logger = logging.getLogger(__name__)

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    logger.info("LangChain not installed; multi-agent mode unavailable")


AGENT_DEFINITIONS: list[dict[str, Any]] = [
    {
        "role": "paper_analyst",
        "display": "PaperAnalyst",
        "prompt": (
            "You are PaperAnalyst, an expert at reading and summarising research papers. "
            "Given paper text, extract the title, authors, abstract, key contributions, "
            "methodology overview, results summary, and limitations. Return structured JSON."
        ),
        "tools": ["parse_paper", "retrieve_context"],
    },
    {
        "role": "concept_extractor",
        "display": "ConceptExtractor",
        "prompt": (
            "You are ConceptExtractor. Given a paper summary, extract every key concept, "
            "method, algorithm name, dataset, metric, and domain-specific term. Return a "
            "structured JSON list of concepts with brief definitions."
        ),
        "tools": ["retrieve_context"],
    },
    {
        "role": "workflow_planner",
        "display": "WorkflowPlanner",
        "prompt": (
            "You are WorkflowPlanner. Given paper analysis and concepts, design a detailed "
            "step-by-step executable workflow that reproduces or applies the paper's methodology. "
            "Each step should specify inputs, outputs, tools needed, and dependencies. "
            "Return a JSON workflow spec."
        ),
        "tools": ["retrieve_context"],
    },
    {
        "role": "code_generator",
        "display": "CodeGenerator",
        "prompt": (
            "You are CodeGenerator. Given a workflow step description and context, produce "
            "clean, well-documented, executable Python code. Include imports, error handling, "
            "type hints, and docstrings. Return the code as a string."
        ),
        "tools": ["generate_code", "retrieve_context"],
    },
    {
        "role": "code_validator",
        "display": "CodeValidator",
        "prompt": (
            "You are CodeValidator. Given generated code, analyse it for correctness, "
            "potential runtime errors, security issues, and style problems. Suggest fixes "
            "where needed. Return a structured validation report."
        ),
        "tools": ["validate_code"],
    },
    {
        "role": "context_retriever",
        "display": "ContextRetriever",
        "prompt": (
            "You are ContextRetriever. Given a query, search the knowledge base for the "
            "most relevant documentation, code examples, and related papers. Summarise and "
            "rank findings by relevance."
        ),
        "tools": ["retrieve_context", "index_document"],
    },
    {
        "role": "dependency_resolver",
        "display": "DependencyResolver",
        "prompt": (
            "You are DependencyResolver. Given a workflow and generated code, identify all "
            "required Python packages, system libraries, and external services. Produce a "
            "pip requirements list with version constraints and installation notes."
        ),
        "tools": ["retrieve_context", "list_repo_structure"],
    },
    {
        "role": "test_generator",
        "display": "TestGenerator",
        "prompt": (
            "You are TestGenerator. Given workflow steps and their code, create comprehensive "
            "pytest test cases covering happy paths, edge cases, and error conditions. "
            "Return executable test code."
        ),
        "tools": ["generate_code", "validate_code"],
    },
    {
        "role": "documentation_writer",
        "display": "DocumentationWriter",
        "prompt": (
            "You are DocumentationWriter. Given the complete workflow, code, and context, "
            "produce clear documentation including a README section, API reference, usage "
            "examples, and architecture notes."
        ),
        "tools": ["retrieve_context"],
    },
    {
        "role": "quality_reviewer",
        "display": "QualityReviewer",
        "prompt": (
            "You are QualityReviewer. Review the entire pipeline output for correctness, "
            "completeness, consistency, and quality. Flag issues, suggest improvements, and "
            "assign a confidence score. Return a structured review report."
        ),
        "tools": ["retrieve_context", "validate_code"],
    },
    {
        "role": "integration_specialist",
        "display": "IntegrationSpecialist",
        "prompt": (
            "You are IntegrationSpecialist. Given workflow outputs, define API specifications "
            "(REST/GraphQL endpoints), data schemas, integration patterns, and deployment "
            "notes. Return structured integration specs."
        ),
        "tools": ["retrieve_context", "list_repo_structure"],
    },
]

AGENT_ROLES = {d["role"]: d for d in AGENT_DEFINITIONS}


class SpecializedAgent:
    """A single specialised agent backed by LangChain's ChatOpenAI."""

    def __init__(
        self,
        role: str,
        system_prompt: str,
        allowed_tools: list[str] | None = None,
        model: str | None = None,
    ):
        self.role = role
        self.system_prompt = system_prompt
        self.allowed_tools = allowed_tools or []
        self.model_name = model or Config.OPENAI_MODEL

        if _HAS_LANGCHAIN:
            self._llm = ChatOpenAI(
                model=self.model_name,
                api_key=Config.OPENAI_API_KEY,
                temperature=0.2,
            )
        else:
            self._llm = None

    def run(self, input_text: str, context: str | None = None) -> dict[str, Any]:
        """Execute this agent on *input_text* and return structured result."""
        if not _HAS_LANGCHAIN or self._llm is None:
            return {
                "role": self.role,
                "output": f"[LangChain unavailable] {input_text[:200]}",
                "tool_calls": [],
                "duration_ms": 0,
            }

        messages = [SystemMessage(content=self.system_prompt)]
        user_content = input_text
        if context:
            user_content = f"Context:\n{context}\n\n---\n\n{input_text}"
        messages.append(HumanMessage(content=user_content))

        t0 = time.time()
        try:
            response = self._llm.invoke(messages)
            output = response.content
            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = [
                    {"name": tc.get("name", ""), "args": tc.get("args", {})}
                    for tc in response.tool_calls
                ]
        except Exception as exc:
            logger.error("Agent %s failed: %s", self.role, exc)
            output = f"Error: {exc}"
            tool_calls = []

        duration_ms = int((time.time() - t0) * 1000)
        return {
            "role": self.role,
            "output": output,
            "tool_calls": tool_calls,
            "duration_ms": duration_ms,
        }


def build_agent(role: str, model: str | None = None) -> SpecializedAgent:
    """Factory: create a SpecializedAgent by role name."""
    defn = AGENT_ROLES.get(role)
    if defn is None:
        raise ValueError(f"Unknown agent role: {role}")
    return SpecializedAgent(
        role=defn["role"],
        system_prompt=defn["prompt"],
        allowed_tools=defn["tools"],
        model=model,
    )


def build_all_agents(model: str | None = None) -> dict[str, SpecializedAgent]:
    """Build every registered agent and return a role→agent mapping."""
    return {defn["role"]: build_agent(defn["role"], model) for defn in AGENT_DEFINITIONS}


def is_langchain_available() -> bool:
    return _HAS_LANGCHAIN

"""Multi-step LLM agent that converts research papers into executable
workflow guidance through iterative tool calling."""

import json
import logging
from typing import Any

from openai import OpenAI

from app.agent.tools import TOOL_DEFINITIONS, execute_tool
from app.agent.prompts import SYSTEM_PROMPT
from config import Config

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates a paper-to-workflow conversion using an LLM with
    tool-calling capabilities.  Each invocation plans its own sequence
    of tool calls, executes them, and returns a structured workflow."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.client = OpenAI(api_key=api_key or Config.OPENAI_API_KEY)
        self.model = model or Config.OPENAI_MODEL
        self.max_steps = Config.MAX_TOOL_CALLS
        self.conversation_history: list[dict] = []

    def run(self, paper_input: str, query: str = "") -> dict[str, Any]:
        """Execute the full paper-to-workflow pipeline.

        Args:
            paper_input: URL, arXiv ID, or raw text of the paper.
            query: Optional user query to focus the extraction.

        Returns:
            Structured workflow dict with steps, code, and recommendations.
        """
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        user_msg = self._build_user_message(paper_input, query)
        self.conversation_history.append({"role": "user", "content": user_msg})

        workflow: dict[str, Any] = {
            "paper_input": paper_input,
            "query": query,
            "steps": [],
            "tool_calls_made": 0,
            "recommendations": [],
            "status": "in_progress",
        }

        for step_idx in range(self.max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )

            message = response.choices[0].message
            self.conversation_history.append(message)

            if not message.tool_calls:
                workflow["final_summary"] = message.content
                workflow["status"] = "complete"
                break

            for tc in message.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                logger.info("Step %d: calling %s(%s)", step_idx + 1, fn_name,
                            list(fn_args.keys()))
                result = execute_tool(fn_name, fn_args)

                workflow["steps"].append({
                    "step": step_idx + 1,
                    "tool": fn_name,
                    "args": fn_args,
                    "result_summary": result.get("summary", ""),
                })
                workflow["tool_calls_made"] += 1

                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })
        else:
            workflow["status"] = "max_steps_reached"

        workflow["recommendations"] = self._extract_recommendations(
            workflow.get("final_summary", "")
        )
        return workflow

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_message(paper_input: str, query: str) -> str:
        parts = [f"Paper / source: {paper_input}"]
        if query:
            parts.append(f"Focus area: {query}")
        parts.append(
            "Analyze this paper and convert it into an executable workflow. "
            "Use the available tools to extract methodology, retrieve "
            "relevant documentation, generate executable code, and "
            "validate the output."
        )
        return "\n\n".join(parts)

    def _extract_recommendations(self, summary: str) -> list[str]:
        if not summary:
            return []
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract actionable recommendations as a JSON "
                            "object with a single key 'recommendations' "
                            "whose value is a list of strings."
                        ),
                    },
                    {"role": "user", "content": summary},
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return data.get("recommendations", [])
        except Exception:
            logger.debug("Recommendation extraction skipped", exc_info=True)
            return []

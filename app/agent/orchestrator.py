"""Multi-agent orchestrator for Paper2Agent.

Supports two execution modes:
  1. **Multi-agent** (LangChain available): runs 11 specialised agents in a
     pipeline with human-in-the-loop checkpoints and full workflow persistence.
  2. **Single-agent** fallback (original): uses raw OpenAI function-calling
     with the same tool definitions as before.

The public interface (``AgentOrchestrator.run``) returns the same workflow
dict in both modes so that ``routes.py`` works unchanged.
"""

import json
import logging
import time
from typing import Any

from openai import OpenAI

from app.agent.tools import TOOL_DEFINITIONS, execute_tool
from app.agent.prompts import SYSTEM_PROMPT
from config import Config

logger = logging.getLogger(__name__)

# Pipeline definition: (role, required?) — order matters
_CORE_PIPELINE = [
    ("paper_analyst", True),
    ("concept_extractor", True),
    ("workflow_planner", True),   # human-in-the-loop checkpoint after this
    ("code_generator", True),     # human-in-the-loop checkpoint after this
    ("code_validator", True),
    ("quality_reviewer", True),
]

_OPTIONAL_PIPELINE = [
    ("context_retriever", False),
    ("dependency_resolver", False),
    ("test_generator", False),
    ("documentation_writer", False),
    ("integration_specialist", False),
]

HUMAN_REVIEW_GATES = {"workflow_planner", "code_generator"}


class AgentOrchestrator:
    """Orchestrates paper-to-workflow conversion.

    Detects LangChain availability at init and selects the appropriate mode.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.client = OpenAI(api_key=api_key or Config.OPENAI_API_KEY)
        self.model = model or Config.OPENAI_MODEL
        self.max_steps = Config.MAX_TOOL_CALLS
        self.conversation_history: list[dict] = []

        self._multi_agent = False
        try:
            from app.agent.agents import is_langchain_available, build_all_agents
            if is_langchain_available():
                self._agents = build_all_agents(self.model)
                self._multi_agent = True
                logger.info("Multi-agent mode enabled (%d agents)", len(self._agents))
        except Exception as exc:
            logger.info("Falling back to single-agent mode: %s", exc)

    def run(self, paper_input: str, query: str = "",
            enable_optional: bool = False) -> dict[str, Any]:
        """Execute the full paper-to-workflow pipeline.

        Returns a workflow dict compatible with the original API contract.
        """
        if self._multi_agent:
            return self._run_multi_agent(paper_input, query, enable_optional)
        return self._run_single_agent(paper_input, query)

    # ------------------------------------------------------------------
    # Multi-agent pipeline
    # ------------------------------------------------------------------

    def _run_multi_agent(self, paper_input: str, query: str,
                         enable_optional: bool) -> dict[str, Any]:
        from app.agent.agents import build_agent

        workflow_dict: dict[str, Any] = {
            "paper_input": paper_input,
            "query": query,
            "steps": [],
            "tool_calls_made": 0,
            "recommendations": [],
            "status": "in_progress",
            "mode": "multi_agent",
            "agent_count": len(_CORE_PIPELINE) + (len(_OPTIONAL_PIPELINE) if enable_optional else 0),
        }

        pipeline = list(_CORE_PIPELINE)
        if enable_optional:
            pipeline.extend(_OPTIONAL_PIPELINE)

        accumulated_context = f"Paper source: {paper_input}"
        if query:
            accumulated_context += f"\nFocus: {query}"

        db_workflow = None
        try:
            from app.db import get_db
            from app.models import Workflow, WorkflowStep, AgentTrace
            with get_db() as session:
                db_workflow = Workflow(
                    source_type="paper",
                    source_ref=paper_input,
                    status="in_progress",
                )
                session.add(db_workflow)
                session.flush()
                workflow_dict["workflow_id"] = db_workflow.id

                for step_order, (role, _required) in enumerate(pipeline):
                    agent = self._agents.get(role) or build_agent(role, self.model)

                    t0 = time.time()
                    result = agent.run(
                        input_text=accumulated_context,
                        context=accumulated_context if step_order > 0 else None,
                    )
                    duration_ms = int((time.time() - t0) * 1000)

                    step_record = WorkflowStep(
                        workflow_id=db_workflow.id,
                        agent_role=role,
                        input_text=accumulated_context[:5000],
                        output_text=result.get("output", "")[:10000],
                        tool_calls=result.get("tool_calls", []),
                        step_order=step_order,
                        duration_ms=duration_ms,
                    )
                    session.add(step_record)
                    session.flush()

                    trace = AgentTrace(
                        workflow_id=db_workflow.id,
                        step_id=step_record.id,
                        agent_role=role,
                        event_type="llm_response",
                        payload={
                            "output_preview": result.get("output", "")[:2000],
                            "tool_calls": result.get("tool_calls", []),
                            "duration_ms": duration_ms,
                        },
                    )
                    session.add(trace)

                    workflow_dict["steps"].append({
                        "step": step_order + 1,
                        "tool": role,
                        "agent_role": role,
                        "result_summary": result.get("output", "")[:500],
                        "duration_ms": duration_ms,
                    })
                    workflow_dict["tool_calls_made"] += 1 + len(result.get("tool_calls", []))

                    accumulated_context += f"\n\n--- {role} output ---\n{result.get('output', '')[:3000]}"

                    if role in HUMAN_REVIEW_GATES:
                        review_trace = AgentTrace(
                            workflow_id=db_workflow.id,
                            step_id=step_record.id,
                            agent_role=role,
                            event_type="human_review",
                            payload={
                                "gate": role,
                                "status": "pending_review",
                                "message": f"Human review checkpoint after {role}",
                            },
                        )
                        session.add(review_trace)

                        workflow_dict.setdefault("pending_reviews", []).append({
                            "gate": role,
                            "step": step_order + 1,
                            "status": "pending_review",
                        })

                db_workflow.summary = accumulated_context[-3000:]
                db_workflow.status = "complete"
                db_workflow.recommendations = self._extract_recommendations(
                    accumulated_context[-3000:]
                )
                workflow_dict["recommendations"] = db_workflow.recommendations
                workflow_dict["final_summary"] = accumulated_context[-3000:]
                workflow_dict["status"] = "complete"

        except ImportError:
            logger.warning("Database modules unavailable; running without persistence")
            for step_order, (role, _required) in enumerate(pipeline):
                agent = self._agents.get(role)
                if agent is None:
                    continue
                result = agent.run(
                    input_text=accumulated_context,
                    context=accumulated_context if step_order > 0 else None,
                )
                workflow_dict["steps"].append({
                    "step": step_order + 1,
                    "tool": role,
                    "agent_role": role,
                    "result_summary": result.get("output", "")[:500],
                    "duration_ms": result.get("duration_ms", 0),
                })
                workflow_dict["tool_calls_made"] += 1
                accumulated_context += f"\n\n--- {role} output ---\n{result.get('output', '')[:3000]}"

            workflow_dict["final_summary"] = accumulated_context[-3000:]
            workflow_dict["status"] = "complete"
            workflow_dict["recommendations"] = self._extract_recommendations(
                accumulated_context[-3000:]
            )

        return workflow_dict

    # ------------------------------------------------------------------
    # Original single-agent mode (backward compatible)
    # ------------------------------------------------------------------

    def _run_single_agent(self, paper_input: str, query: str) -> dict[str, Any]:
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
            "mode": "single_agent",
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

        self._persist_single_agent_workflow(workflow)
        return workflow

    def _persist_single_agent_workflow(self, workflow: dict):
        """Best-effort persistence for single-agent runs."""
        try:
            from app.db import get_db
            from app.models import Workflow as WfModel

            with get_db() as session:
                wf = WfModel(
                    source_type="paper",
                    source_ref=workflow.get("paper_input", ""),
                    summary=workflow.get("final_summary", ""),
                    recommendations=workflow.get("recommendations", []),
                    status=workflow.get("status", "complete"),
                )
                session.add(wf)
                session.flush()
                workflow["workflow_id"] = wf.id
        except Exception:
            logger.debug("Workflow persistence skipped", exc_info=True)

    # ------------------------------------------------------------------
    # Helpers
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

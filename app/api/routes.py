"""REST API endpoints for Paper2Agent.

Routes
------
GET  /              Web UI
POST /api/convert   Convert a paper into an executable workflow
GET  /api/workflows         List all saved workflows (from DB)
GET  /api/workflows/<id>    Get a specific workflow with steps and traces
GET  /api/workflows/<id>/traces  Get traces for a workflow
POST /api/workflows/<id>/approve Approve a pending human-review step
POST /api/search    Semantic search over the knowledge base
POST /api/index     Add a document to the knowledge base
GET  /health        Liveness probe
"""

import logging

from flask import Blueprint, jsonify, render_template, request

from ariadne import graphql_sync
from app.agent.orchestrator import AgentOrchestrator
from app.agent.tools import get_shared_rag
from app.api.graphql_schema import schema

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)

_orchestrator: AgentOrchestrator | None = None


def _get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


# ── Pages ────────────────────────────────────────────────────────────

@api_bp.route("/")
def index():
    return render_template("index.html")


# ── Core conversion ─────────────────────────────────────────────────

@api_bp.route("/api/convert", methods=["POST"])
def convert_paper():
    """Accept a paper source (URL / arXiv ID / text) and return a
    structured, executable workflow."""
    data = request.get_json(force=True)
    source = data.get("source", "")
    query = data.get("query", "")

    if not source:
        return jsonify({"error": "Missing 'source' field"}), 400

    try:
        result = _get_orchestrator().run(paper_input=source, query=query)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Conversion failed")
        return jsonify({"error": str(exc)}), 500


# ── Workflow persistence endpoints ──────────────────────────────────

@api_bp.route("/api/workflows", methods=["GET"])
def list_workflows():
    """List all saved workflows from the database."""
    try:
        from app.db import get_db
        from app.models import Workflow

        with get_db() as session:
            workflows = session.query(Workflow).order_by(
                Workflow.created_at.desc()
            ).all()
            return jsonify({
                "workflows": [w.to_dict() for w in workflows],
                "count": len(workflows),
            })
    except Exception:
        logger.debug("DB unavailable, falling back to RAG-based listing", exc_info=True)
        rag = get_shared_rag()
        grouped: dict[str, dict] = {}
        for doc in rag.documents:
            did = doc.get("doc_id", "unknown")
            if did not in grouped:
                grouped[did] = {
                    "doc_id": did,
                    "source": doc.get("metadata", {}).get("source", ""),
                    "chunks": 0,
                }
            grouped[did]["chunks"] += 1
        return jsonify({"workflows": list(grouped.values())})


@api_bp.route("/api/workflows/<workflow_id>", methods=["GET"])
def get_workflow(workflow_id: str):
    """Get a specific workflow with its steps and traces."""
    try:
        from app.db import get_db
        from app.models import Workflow

        with get_db() as session:
            workflow = session.query(Workflow).filter_by(id=workflow_id).first()
            if workflow is None:
                return jsonify({"error": "Workflow not found"}), 404

            result = workflow.to_dict()
            result["traces"] = [t.to_dict() for t in workflow.traces]
            return jsonify(result)
    except ImportError:
        return jsonify({"error": "Database not configured"}), 501


@api_bp.route("/api/workflows/<workflow_id>/traces", methods=["GET"])
def get_workflow_traces(workflow_id: str):
    """Get all agent traces for a workflow."""
    try:
        from app.db import get_db
        from app.models import AgentTrace

        with get_db() as session:
            traces = (
                session.query(AgentTrace)
                .filter_by(workflow_id=workflow_id)
                .order_by(AgentTrace.timestamp)
                .all()
            )
            return jsonify({
                "workflow_id": workflow_id,
                "traces": [t.to_dict() for t in traces],
                "count": len(traces),
            })
    except ImportError:
        return jsonify({"error": "Database not configured"}), 501


@api_bp.route("/api/workflows/<workflow_id>/approve", methods=["POST"])
def approve_workflow_step(workflow_id: str):
    """Approve a pending human-in-the-loop review step."""
    data = request.get_json(force=True)
    gate = data.get("gate", "")
    approved = data.get("approved", True)
    comment = data.get("comment", "")

    try:
        from app.db import get_db
        from app.models import AgentTrace, Workflow

        with get_db() as session:
            workflow = session.query(Workflow).filter_by(id=workflow_id).first()
            if workflow is None:
                return jsonify({"error": "Workflow not found"}), 404

            review_traces = (
                session.query(AgentTrace)
                .filter_by(
                    workflow_id=workflow_id,
                    event_type="human_review",
                )
                .all()
            )

            matched = None
            for t in review_traces:
                payload = t.payload or {}
                if payload.get("gate") == gate and payload.get("status") == "pending_review":
                    matched = t
                    break

            if matched is None:
                return jsonify({"error": f"No pending review found for gate '{gate}'"}), 404

            matched.payload = {
                **matched.payload,
                "status": "approved" if approved else "rejected",
                "comment": comment,
                "reviewed": True,
            }
            matched.event_type = "human_review"

            return jsonify({
                "workflow_id": workflow_id,
                "gate": gate,
                "status": "approved" if approved else "rejected",
                "message": f"Review for '{gate}' recorded",
            })
    except ImportError:
        return jsonify({"error": "Database not configured"}), 501


# ── Knowledge base ──────────────────────────────────────────────────

@api_bp.route("/api/search", methods=["POST"])
def search_knowledge():
    data = request.get_json(force=True)
    query = data.get("query", "")
    top_k = data.get("top_k", 5)

    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400

    results = get_shared_rag().search(query, top_k=top_k)
    return jsonify({"results": results, "count": len(results)})


@api_bp.route("/api/index", methods=["POST"])
def index_document():
    data = request.get_json(force=True)
    text = data.get("text", "")
    metadata = data.get("metadata", {})

    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    doc_id = get_shared_rag().add_document(text, metadata=metadata)
    return jsonify({"doc_id": doc_id, "status": "indexed"})


# ── GraphQL ──────────────────────────────────────────────────────────

@api_bp.route("/graphql", methods=["POST"])
def graphql_endpoint():
    data = request.get_json(force=True)
    success, result = graphql_sync(schema, data, context_value=request)
    return jsonify(result), 200 if success else 400


# ── Health ───────────────────────────────────────────────────────────

@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "paper2agent"})

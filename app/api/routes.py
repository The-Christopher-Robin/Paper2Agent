"""REST API endpoints for Paper2Agent.

Routes
------
GET  /              Web UI
POST /api/convert   Convert a paper into an executable workflow
GET  /api/workflows List indexed documents / workflows
POST /api/search    Semantic search over the knowledge base
POST /api/index     Add a document to the knowledge base
GET  /health        Liveness probe
"""

import logging

from flask import Blueprint, jsonify, render_template, request

from app.agent.orchestrator import AgentOrchestrator
from app.retrieval.rag import RAGRetriever

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)

_orchestrator: AgentOrchestrator | None = None
_rag: RAGRetriever | None = None


def _get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


def _get_rag() -> RAGRetriever:
    global _rag
    if _rag is None:
        _rag = RAGRetriever()
    return _rag


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


# ── Knowledge base ──────────────────────────────────────────────────

@api_bp.route("/api/workflows", methods=["GET"])
def list_workflows():
    rag = _get_rag()
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


@api_bp.route("/api/search", methods=["POST"])
def search_knowledge():
    data = request.get_json(force=True)
    query = data.get("query", "")
    top_k = data.get("top_k", 5)

    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400

    results = _get_rag().search(query, top_k=top_k)
    return jsonify({"results": results, "count": len(results)})


@api_bp.route("/api/index", methods=["POST"])
def index_document():
    data = request.get_json(force=True)
    text = data.get("text", "")
    metadata = data.get("metadata", {})

    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    doc_id = _get_rag().add_document(text, metadata=metadata)
    return jsonify({"doc_id": doc_id, "status": "indexed"})


# ── Health ───────────────────────────────────────────────────────────

@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "paper2agent"})

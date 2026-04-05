"""GraphQL schema for Paper2Agent.

Exposes the same functionality as the REST API through a single
``/graphql`` endpoint, supporting queries for workflows and semantic
search, and mutations for paper conversion and document indexing.
"""

from ariadne import (
    QueryType,
    MutationType,
    make_executable_schema,
)

from app.agent.tools import get_shared_rag

type_defs = """
    type Query {
        workflows: [Workflow!]!
        workflow(id: String!): WorkflowDetail
        search(query: String!, topK: Int = 5): [SearchResult!]!
        health: HealthStatus!
    }

    type Mutation {
        convertPaper(source: String!, query: String): ConversionResult!
        indexDocument(text: String!, source: String): IndexResult!
    }

    type Workflow {
        docId: String!
        source: String
        chunks: Int!
        status: String
        createdAt: String
    }

    type WorkflowDetail {
        id: String!
        sourceType: String
        sourceRef: String
        summary: String
        status: String
        createdAt: String
        updatedAt: String
        steps: [PipelineStep!]!
        traces: [AgentTraceType!]!
    }

    type PipelineStep {
        id: String!
        agentRole: String!
        outputText: String
        stepOrder: Int!
        durationMs: Int
    }

    type AgentTraceType {
        id: String!
        agentRole: String!
        eventType: String!
        timestamp: String
    }

    type SearchResult {
        text: String!
        score: Float!
        section: String
        source: String
    }

    type ConversionResult {
        status: String!
        toolCallsMade: Int!
        steps: [WorkflowStep!]!
        finalSummary: String
    }

    type WorkflowStep {
        step: Int!
        tool: String!
        resultSummary: String
    }

    type IndexResult {
        docId: String!
        status: String!
    }

    type HealthStatus {
        status: String!
        service: String!
    }
"""

query = QueryType()
mutation = MutationType()


@query.field("workflows")
def resolve_workflows(*_):
    try:
        from app.db import get_db
        from app.models import Workflow

        with get_db() as session:
            workflows = session.query(Workflow).order_by(
                Workflow.created_at.desc()
            ).all()
            return [
                {
                    "docId": w.id,
                    "source": w.source_ref,
                    "chunks": len(w.steps),
                    "status": w.status,
                    "createdAt": w.created_at.isoformat() if w.created_at else None,
                }
                for w in workflows
            ]
    except Exception:
        rag = get_shared_rag()
        grouped: dict[str, dict] = {}
        for doc in rag.documents:
            did = doc.get("doc_id", "unknown")
            if did not in grouped:
                grouped[did] = {
                    "docId": did,
                    "source": doc.get("metadata", {}).get("source", ""),
                    "chunks": 0,
                }
            grouped[did]["chunks"] += 1
        return list(grouped.values())


@query.field("workflow")
def resolve_workflow(*_, id: str):
    try:
        from app.db import get_db
        from app.models import Workflow

        with get_db() as session:
            w = session.query(Workflow).filter_by(id=id).first()
            if w is None:
                return None
            return {
                "id": w.id,
                "sourceType": w.source_type,
                "sourceRef": w.source_ref,
                "summary": w.summary,
                "status": w.status,
                "createdAt": w.created_at.isoformat() if w.created_at else None,
                "updatedAt": w.updated_at.isoformat() if w.updated_at else None,
                "steps": [
                    {
                        "id": s.id,
                        "agentRole": s.agent_role,
                        "outputText": s.output_text,
                        "stepOrder": s.step_order,
                        "durationMs": s.duration_ms,
                    }
                    for s in w.steps
                ],
                "traces": [
                    {
                        "id": t.id,
                        "agentRole": t.agent_role,
                        "eventType": t.event_type,
                        "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                    }
                    for t in w.traces
                ],
            }
    except Exception:
        return None


@query.field("search")
def resolve_search(*_, query: str, topK: int = 5):
    results = get_shared_rag().search(query, top_k=topK)
    return [
        {
            "text": r["text"],
            "score": r.get("score", 0.0),
            "section": r.get("metadata", {}).get("section"),
            "source": r.get("metadata", {}).get("source"),
        }
        for r in results
    ]


@query.field("health")
def resolve_health(*_):
    return {"status": "healthy", "service": "paper2agent"}


@mutation.field("convertPaper")
def resolve_convert_paper(*_, source: str, query: str = ""):
    from app.agent.orchestrator import AgentOrchestrator

    orch = AgentOrchestrator()
    result = orch.run(paper_input=source, query=query or "")
    return {
        "status": result.get("status", "unknown"),
        "toolCallsMade": result.get("tool_calls_made", 0),
        "steps": [
            {
                "step": s.get("step", 0),
                "tool": s.get("tool", ""),
                "resultSummary": s.get("result_summary", ""),
            }
            for s in result.get("steps", [])
        ],
        "finalSummary": result.get("final_summary"),
    }


@mutation.field("indexDocument")
def resolve_index_document(*_, text: str, source: str = ""):
    doc_id = get_shared_rag().add_document(text, metadata={"source": source})
    return {"docId": doc_id, "status": "indexed"}


schema = make_executable_schema(type_defs, query, mutation)

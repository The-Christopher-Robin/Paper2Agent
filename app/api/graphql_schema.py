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

from app.retrieval.rag import RAGRetriever

type_defs = """
    type Query {
        workflows: [Workflow!]!
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

_rag = None


def _get_rag() -> RAGRetriever:
    global _rag
    if _rag is None:
        _rag = RAGRetriever()
    return _rag


@query.field("workflows")
def resolve_workflows(*_):
    rag = _get_rag()
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


@query.field("search")
def resolve_search(*_, query: str, topK: int = 5):
    results = _get_rag().search(query, top_k=topK)
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
                "step": s["step"],
                "tool": s["tool"],
                "resultSummary": s.get("result_summary", ""),
            }
            for s in result.get("steps", [])
        ],
        "finalSummary": result.get("final_summary"),
    }


@mutation.field("indexDocument")
def resolve_index_document(*_, text: str, source: str = ""):
    doc_id = _get_rag().add_document(text, metadata={"source": source})
    return {"docId": doc_id, "status": "indexed"}


schema = make_executable_schema(type_defs, query, mutation)

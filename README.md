# Paper2Agent

[![CI](https://github.com/The-Christopher-Robin/Paper2Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/The-Christopher-Robin/Paper2Agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-purple.svg)](https://docs.astral.sh/ruff/)

A multi-agent LLM system that converts research papers into executable workflow guidance using LangChain, OpenAI GPT-4, retrieval-augmented generation (FAISS + pgvector), human-in-the-loop oversight, and a unified REST/GraphQL API with PostgreSQL-backed persistence.

---

## Overview

Paper2Agent takes a paper source — arXiv ID, GitHub repository URL, or raw text — and runs an **11-agent pipeline** that:

1. **PaperAnalyst** parses the paper into structured sections (abstract, methodology, results)
2. **ConceptExtractor** extracts key concepts, methods, algorithms, and terminology
3. **WorkflowPlanner** designs a step-by-step executable workflow → *human review checkpoint*
4. **CodeGenerator** generates executable Python code for each step → *human review checkpoint*
5. **CodeValidator** validates generated code in a sandboxed environment
6. **QualityReviewer** reviews the complete output for correctness and completeness

Optional agents (configurable):
- **ContextRetriever** — searches the knowledge base for relevant documentation
- **DependencyResolver** — identifies required packages and system dependencies
- **TestGenerator** — creates pytest test cases for workflow steps
- **DocumentationWriter** — produces README sections, API docs, and usage examples
- **IntegrationSpecialist** — defines API specs and integration patterns

The system exposes both REST and GraphQL APIs, persists workflows and agent traces in PostgreSQL, and includes a web interface for interactive use.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Web UI                                │
│                  (templates/index.html)                       │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                  Flask API Layer                              │
│                                                               │
│   REST                              GraphQL (Ariadne)         │
│   POST /api/convert                 mutation convertPaper     │
│   GET  /api/workflows               query workflows           │
│   GET  /api/workflows/<id>          query workflow(id)         │
│   GET  /api/workflows/<id>/traces                             │
│   POST /api/workflows/<id>/approve                            │
│   POST /api/search                  query search              │
│   POST /api/index                   mutation indexDocument     │
└───────┬──────────────────────────────────┬────────────────────┘
        │                                  │
┌───────▼─────────────────┐        ┌───────▼────────────────────┐
│  Multi-Agent            │        │  Dual RAG Retriever         │
│  Orchestrator           │◄──────►│                             │
│  (LangChain pipeline)   │        │  FAISS (in-memory/disk)     │
│                         │        │  pgvector (PostgreSQL)      │
│  11 Specialized Agents  │        │  OpenAI Embeddings          │
│  Human-in-the-loop      │        └─────────────────────────────┘
│  Workflow persistence   │
└───────┬─────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────┐
│  Tool Suite                                                   │
│                                                               │
│  parse_paper        retrieve_context     generate_code        │
│  validate_code      index_document       list_repo_structure  │
└───────┬──────────────────────────────────────────────────────┘
        │
┌───────▼───────────────┐    ┌─────────────────────────────────┐
│  Paper Parser         │    │  PostgreSQL + pgvector           │
│  arXiv · GitHub       │    │  Workflows · Steps · Traces      │
│  PDF · raw text       │    │  Documents with vector embeddings │
└───────────────────────┘    └─────────────────────────────────┘
```

### Agent Pipeline

The orchestrator runs agents sequentially, passing accumulated context through the pipeline:

```
PaperAnalyst → ConceptExtractor → WorkflowPlanner → [Human Review]
    → CodeGenerator → [Human Review] → CodeValidator → QualityReviewer
```

Each agent:
- Uses `langchain_openai.ChatOpenAI` with a role-specific system prompt
- Returns structured output with `{role, output, tool_calls, duration_ms}`
- Gets logged as a `WorkflowStep` with an `AgentTrace` in PostgreSQL

**Human-in-the-loop:** After `WorkflowPlanner` and `CodeGenerator`, the system creates review checkpoints. These can be approved or rejected via `POST /api/workflows/<id>/approve`.

### Fallback Mode

If LangChain is not installed, the orchestrator falls back to the original single-agent OpenAI function-calling loop with the same 6 tools, preserving backward compatibility.

## Quick Start

### Prerequisites

- Python 3.10+
- An OpenAI API key
- PostgreSQL with pgvector extension (optional — SQLite used by default)

### Install

```bash
git clone https://github.com/The-Christopher-Robin/Paper2Agent.git
cd Paper2Agent
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
# optionally set DATABASE_URL for PostgreSQL
```

### Run

```bash
python run.py
```

Open **http://localhost:5000** in your browser.

### Docker (with PostgreSQL + pgvector)

```bash
docker compose up --build
```

This starts both the Paper2Agent Flask app and a PostgreSQL 16 instance with pgvector pre-installed. The app automatically connects to the database and creates all required tables.

Or manually without Docker Compose:

```bash
docker build -t paper2agent .
docker run -p 5000:5000 -e OPENAI_API_KEY=sk-... paper2agent
```

## API Reference

### REST

| Method | Endpoint                          | Description                                      |
|--------|-----------------------------------|--------------------------------------------------|
| POST   | `/api/convert`                    | Convert a paper into an executable workflow       |
| GET    | `/api/workflows`                  | List all saved workflows from the database        |
| GET    | `/api/workflows/<id>`             | Get a workflow with its steps and traces          |
| GET    | `/api/workflows/<id>/traces`      | Get all agent traces for a workflow               |
| POST   | `/api/workflows/<id>/approve`     | Approve/reject a human-in-the-loop review step    |
| POST   | `/api/search`                     | Semantic search over the knowledge base           |
| POST   | `/api/index`                      | Add a document to the knowledge base              |
| GET    | `/health`                         | Liveness probe                                    |

#### Convert a paper

```bash
curl -X POST http://localhost:5000/api/convert \
  -H "Content-Type: application/json" \
  -d '{"source": "2301.12345", "query": "extract the training pipeline"}'
```

#### List workflows

```bash
curl http://localhost:5000/api/workflows
```

#### Get workflow details with traces

```bash
curl http://localhost:5000/api/workflows/<workflow-id>
```

#### Approve a human-review checkpoint

```bash
curl -X POST http://localhost:5000/api/workflows/<id>/approve \
  -H "Content-Type: application/json" \
  -d '{"gate": "workflow_planner", "approved": true, "comment": "Plan looks good"}'
```

#### Search the knowledge base

```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "attention mechanism", "top_k": 5}'
```

### GraphQL

Single endpoint at `POST /graphql`.

```bash
curl -X POST http://localhost:5000/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ search(query: \"transformer\", topK: 3) { text score section } }"}'
```

**Available operations:**

| Type     | Field            | Arguments                         |
|----------|------------------|-----------------------------------|
| Query    | `workflows`      | —                                 |
| Query    | `workflow`       | `id: String!`                     |
| Query    | `search`         | `query: String!, topK: Int`       |
| Query    | `health`         | —                                 |
| Mutation | `convertPaper`   | `source: String!, query: String`  |
| Mutation | `indexDocument`  | `text: String!, source: String`   |

## Data Model

### Workflow Persistence

All workflows, agent steps, and traces are persisted in PostgreSQL (or SQLite for local development):

- **Workflow** — top-level record with status, source, summary, and JSON recommendations
- **WorkflowStep** — one per agent execution, capturing input/output, tool calls, timing
- **AgentTrace** — fine-grained event log (tool_call, llm_response, human_review, error)
- **Document** — indexed document chunks with optional pgvector embeddings

JSON-serialized workflow specs and agent traces enable full replay and debugging of any pipeline execution.

### RAG (Retrieval-Augmented Generation)

Two retrieval backends work in parallel:
- **FAISS** — in-memory approximate nearest-neighbour search (default, always available)
- **pgvector** — PostgreSQL-native vector similarity search (used when PostgreSQL is configured)

Both use OpenAI `text-embedding-3-small` embeddings (1536-dimensional) with a deterministic hash-based fallback for offline/test use.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Project Structure

```
paper2agent/
├── run.py                          # Entrypoint
├── config.py                       # Environment-based configuration (incl. DATABASE_URL)
├── pyproject.toml                  # Project metadata, ruff + pytest config
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container image
├── docker-compose.yml              # Compose stack (app + PostgreSQL/pgvector)
├── .env.example                    # Environment variable template
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI (lint + test)
├── app/
│   ├── __init__.py                 # Flask app factory (with DB init)
│   ├── db.py                       # SQLAlchemy engine, session, and table creation
│   ├── models.py                   # ORM models (Workflow, WorkflowStep, AgentTrace, Document)
│   ├── agent/
│   │   ├── orchestrator.py         # Multi-agent pipeline orchestrator (LangChain + fallback)
│   │   ├── agents.py               # 11 specialized LangChain agent definitions
│   │   ├── tools.py                # Tool definitions, dispatch, shared RAG singleton
│   │   └── prompts.py              # System and planning prompts
│   ├── retrieval/
│   │   ├── rag.py                  # FAISS-backed RAG retriever
│   │   └── pgvector_store.py       # pgvector-backed retriever for PostgreSQL
│   ├── parsers/
│   │   └── paper_parser.py         # arXiv, GitHub, file, and text parsers
│   └── api/
│       ├── routes.py               # REST API endpoints (incl. workflow + approval)
│       └── graphql_schema.py       # GraphQL schema and resolvers
├── templates/
│   └── index.html                  # Web interface
└── tests/
    └── test_agent.py               # Smoke tests
```

## Tech Stack

| Layer         | Technology                                            |
|---------------|-------------------------------------------------------|
| Backend       | Python, Flask, Gunicorn                               |
| LLM Framework | LangChain, langchain-openai                           |
| LLM           | OpenAI API (GPT-4, function calling)                  |
| Agents        | 11 specialized LangChain agents with role prompts     |
| Retrieval     | FAISS + pgvector, OpenAI Embeddings                   |
| Database      | PostgreSQL (pgvector), SQLAlchemy ORM, SQLite fallback|
| API           | REST + GraphQL (Ariadne)                              |
| Frontend      | Vanilla HTML/CSS/JS                                   |
| Infra         | Docker, Docker Compose, GitHub Actions CI             |
| Testing       | pytest, ruff                                          |

## License

[MIT](LICENSE)

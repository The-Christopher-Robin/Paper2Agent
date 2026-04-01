# Paper2Agent

[![CI](https://github.com/The-Christopher-Robin/Paper2Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/The-Christopher-Robin/Paper2Agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-purple.svg)](https://docs.astral.sh/ruff/)

An agent-based LLM copilot that converts research papers into executable workflow guidance through multi-step tool calling, retrieval-augmented generation, and a unified REST/GraphQL API.

---

## Overview

Paper2Agent takes a paper source — arXiv ID, GitHub repository URL, or raw text — and runs a multi-step agent loop that:

1. **Parses** the paper into structured sections (abstract, methodology, results)
2. **Indexes** extracted content into a FAISS-backed vector store
3. **Retrieves** semantically relevant documentation and code examples via RAG
4. **Generates** step-by-step executable workflows with code
5. **Validates** generated code in a sandboxed environment

The system exposes both REST and GraphQL APIs, and includes a web interface for interactive use.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   Web UI                         │
│             (templates/index.html)               │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│           Flask API Layer                        │
│                                                  │
│   REST                        GraphQL            │
│   POST /api/convert           mutation convert   │
│   POST /api/search            query search       │
│   POST /api/index             mutation index     │
│   GET  /api/workflows         query workflows    │
└───────┬──────────────────────────┬───────────────┘
        │                          │
┌───────▼──────────┐      ┌───────▼────────────────┐
│  Agent           │      │  RAG Retriever         │
│  Orchestrator    │◄────►│  FAISS + embeddings    │
│                  │      │  (OpenAI / offline)    │
│  • plan          │      └────────────────────────┘
│  • tool dispatch │
│  • summarize     │
└───────┬──────────┘
        │
┌───────▼──────────────────────────────────────────┐
│  Tool Suite                                      │
│                                                  │
│  parse_paper        retrieve_context             │
│  generate_code      validate_code                │
│  index_document     list_repo_structure           │
└───────┬──────────────────────────────────────────┘
        │
┌───────▼──────────┐
│  Paper Parser    │
│  arXiv · GitHub  │
│  PDF · raw text  │
└──────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- An OpenAI API key

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
```

### Run

```bash
python run.py
```

Open **http://localhost:5000** in your browser.

### Docker

```bash
docker compose up --build
```

Or manually:

```bash
docker build -t paper2agent .
docker run -p 5000:5000 -e OPENAI_API_KEY=sk-... paper2agent
```

## API Reference

### REST

| Method | Endpoint          | Description                                |
|--------|-------------------|--------------------------------------------|
| POST   | `/api/convert`    | Convert a paper into an executable workflow |
| GET    | `/api/workflows`  | List all indexed documents                 |
| POST   | `/api/search`     | Semantic search over the knowledge base    |
| POST   | `/api/index`      | Add a document to the knowledge base       |
| GET    | `/health`         | Liveness probe                             |

#### Convert a paper

```bash
curl -X POST http://localhost:5000/api/convert \
  -H "Content-Type: application/json" \
  -d '{"source": "2301.12345", "query": "extract the training pipeline"}'
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

| Type     | Field            | Arguments                    |
|----------|------------------|------------------------------|
| Query    | `workflows`      | —                            |
| Query    | `search`         | `query: String!, topK: Int`  |
| Query    | `health`         | —                            |
| Mutation | `convertPaper`   | `source: String!, query: String` |
| Mutation | `indexDocument`  | `text: String!, source: String`  |

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Project Structure

```
paper2agent/
├── run.py                        # Entrypoint
├── config.py                     # Environment-based configuration
├── pyproject.toml                # Project metadata, ruff + pytest config
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container image
├── docker-compose.yml            # Compose stack
├── .env.example                  # Environment variable template
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI (lint + test)
├── app/
│   ├── __init__.py               # Flask app factory
│   ├── agent/
│   │   ├── orchestrator.py       # Multi-step LLM agent with tool calling
│   │   ├── tools.py              # Tool definitions and dispatch handlers
│   │   └── prompts.py            # System and planning prompts
│   ├── retrieval/
│   │   └── rag.py                # FAISS-backed RAG retriever
│   ├── parsers/
│   │   └── paper_parser.py       # arXiv, GitHub, file, and text parsers
│   └── api/
│       ├── routes.py             # REST API endpoints
│       └── graphql_schema.py     # GraphQL schema and resolvers
├── templates/
│   └── index.html                # Web interface
└── tests/
    └── test_agent.py             # Smoke tests
```

## Tech Stack

| Layer       | Technology                              |
|-------------|-----------------------------------------|
| Backend     | Python, Flask, Gunicorn                 |
| LLM         | OpenAI API (GPT-4, tool calling)        |
| Retrieval   | FAISS, OpenAI Embeddings                |
| API         | REST + GraphQL (Ariadne)                |
| Frontend    | Vanilla HTML/CSS/JS                     |
| Infra       | Docker, GitHub Actions CI               |
| Testing     | pytest, ruff                            |

## License

[MIT](LICENSE)

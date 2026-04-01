# Paper2Agent

An agent-based LLM copilot that converts research papers into executable workflow guidance.

Paper2Agent takes a paper source (arXiv ID, GitHub URL, or raw text), parses its structure, and uses a multi-step tool-calling agent to produce step-by-step executable workflows—complete with code, dependencies, and recommendations.

Inspired by [jmiao24/Paper2Agent](https://github.com/jmiao24/Paper2Agent).

## Architecture

```
┌─────────────────────────────────────────────┐
│                  Web UI                     │
│            (templates/index.html)           │
└──────────────────┬──────────────────────────┘
                   │  REST API
┌──────────────────▼──────────────────────────┐
│              Flask API Layer                │
│           (app/api/routes.py)               │
│   POST /api/convert  GET /api/workflows     │
│   POST /api/search   POST /api/index        │
└──────┬───────────────────────┬──────────────┘
       │                       │
┌──────▼──────────┐   ┌───────▼──────────────┐
│  Agent          │   │  RAG Retriever       │
│  Orchestrator   │◄──┤  (FAISS + OpenAI     │
│  (multi-step    │   │   embeddings)        │
│   tool calling) │   └──────────────────────┘
└──────┬──────────┘
       │  calls tools
┌──────▼──────────────────────────────────────┐
│  Tools                                      │
│  • parse_paper     • retrieve_context       │
│  • generate_code   • validate_code          │
│  • index_document  • list_repo_structure    │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────┐
│  Paper Parser   │
│  (arXiv, GitHub,│
│   PDF, text)    │
└─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- An OpenAI API key (set as `OPENAI_API_KEY` env var)

### Installation

```bash
git clone https://github.com/The-Christopher-Robin/paper2agent.git
cd paper2agent
pip install -r requirements.txt
```

### Run

```bash
export OPENAI_API_KEY=your-key-here
python run.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### Docker

```bash
docker build -t paper2agent .
docker run -p 5000:5000 -e OPENAI_API_KEY=your-key paper2agent
```

## API

| Method | Endpoint          | Description                              |
|--------|-------------------|------------------------------------------|
| POST   | `/api/convert`    | Convert a paper to an executable workflow |
| GET    | `/api/workflows`  | List indexed documents                   |
| POST   | `/api/search`     | Semantic search over the knowledge base  |
| POST   | `/api/index`      | Add a document to the knowledge base     |
| POST   | `/graphql`        | GraphQL endpoint (queries + mutations)   |
| GET    | `/health`         | Liveness probe                           |

### Convert a paper

```bash
curl -X POST http://localhost:5000/api/convert \
  -H "Content-Type: application/json" \
  -d '{"source": "2301.12345", "query": "extract training pipeline"}'
```

### Search the knowledge base

```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "attention mechanism", "top_k": 5}'
```

### GraphQL

```bash
curl -X POST http://localhost:5000/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ search(query: \"attention\", topK: 3) { text score section } }"}'
```

## Tests

```bash
pip install pytest
pytest tests/ -v
```

## Project Structure

```
paper2agent/
├── run.py                   # Entrypoint
├── config.py                # Configuration
├── Dockerfile
├── requirements.txt
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── agent/
│   │   ├── orchestrator.py  # Multi-step LLM agent
│   │   ├── tools.py         # Tool definitions & handlers
│   │   └── prompts.py       # System prompts
│   ├── retrieval/
│   │   └── rag.py           # FAISS-backed RAG retriever
│   ├── parsers/
│   │   └── paper_parser.py  # arXiv / GitHub / file parsers
│   └── api/
│       ├── routes.py        # REST endpoints
│       └── graphql_schema.py # GraphQL schema & resolvers
├── templates/
│   └── index.html           # Web interface
└── tests/
    └── test_agent.py        # Smoke tests
```

## License

MIT

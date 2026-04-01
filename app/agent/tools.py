"""Tool definitions and execution handlers used by the agent orchestrator.

Each tool follows the OpenAI function-calling schema and has a matching
handler that performs the actual work (parsing, retrieval, code gen, etc.).
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy singletons so we don't import heavy modules at startup.
_rag = None


def _get_rag():
    global _rag
    if _rag is None:
        from app.retrieval.rag import RAGRetriever
        _rag = RAGRetriever()
    return _rag


# ── OpenAI function-calling schemas ──────────────────────────────────

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "parse_paper",
            "description": (
                "Parse a research paper from a URL, arXiv ID, or file path "
                "and extract structured sections (abstract, methodology, "
                "results, references)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "URL, arXiv ID, or local path",
                    },
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_context",
            "description": (
                "Search the knowledge base for documentation, code examples, "
                "or related work using semantic similarity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_code",
            "description": (
                "Generate executable code for a specific methodology step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "What the code should accomplish",
                    },
                    "context": {
                        "type": "string",
                        "description": "Relevant context from paper or docs",
                    },
                    "language": {
                        "type": "string",
                        "enum": ["python", "bash", "r"],
                        "default": "python",
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_code",
            "description": (
                "Run generated code in a sandboxed environment and report "
                "whether it executes successfully."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to validate",
                    },
                    "expected_behavior": {
                        "type": "string",
                        "description": "Description of expected output",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "index_document",
            "description": (
                "Add a document chunk to the retrieval knowledge base "
                "for later semantic search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text content to index",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata dict",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_repo_structure",
            "description": (
                "List the file tree of a cloned repository to understand "
                "its layout before extracting tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Local path to the repository",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Max directory depth",
                        "default": 3,
                    },
                },
                "required": ["repo_path"],
            },
        },
    },
]


# ── Dispatch ─────────────────────────────────────────────────────────

_HANDLERS: dict[str, callable] = {}


def _register(name: str):
    def decorator(fn):
        _HANDLERS[name] = fn
        return fn
    return decorator


def execute_tool(name: str, args: dict) -> dict[str, Any]:
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}",
                "summary": f"Tool '{name}' not found"}
    try:
        return handler(**args)
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return {"error": str(exc),
                "summary": f"Tool '{name}' error: {exc}"}


# ── Handler implementations ──────────────────────────────────────────

@_register("parse_paper")
def _handle_parse_paper(source: str) -> dict:
    from app.parsers.paper_parser import PaperParser

    parser = PaperParser()
    result = parser.parse(source)

    rag = _get_rag()
    for section, content in result.get("sections", {}).items():
        if content:
            rag.add_document(content,
                             metadata={"source": source, "section": section})

    return {
        "title": result.get("title", ""),
        "sections": list(result.get("sections", {}).keys()),
        "abstract": result.get("sections", {}).get("abstract", "")[:500],
        "summary": (f"Parsed paper '{result.get('title', source)}', "
                     f"found {len(result.get('sections', {}))} sections"),
    }


@_register("retrieve_context")
def _handle_retrieve_context(query: str, top_k: int = 5) -> dict:
    results = _get_rag().search(query, top_k=top_k)
    return {
        "results": results,
        "count": len(results),
        "summary": f"Found {len(results)} relevant passages for: {query[:80]}",
    }


@_register("generate_code")
def _handle_generate_code(description: str, context: str = "",
                           language: str = "python") -> dict:
    return {
        "description": description,
        "language": language,
        "context_provided": bool(context),
        "summary": f"Code generation request ({language}): {description[:80]}",
    }


@_register("validate_code")
def _handle_validate_code(code: str, expected_behavior: str = "") -> dict:
    result: dict[str, Any] = {
        "code_length": len(code),
        "syntax_valid": False,
        "execution": None,
    }

    try:
        compile(code, "<validate>", "exec")
        result["syntax_valid"] = True
    except SyntaxError as exc:
        result["syntax_error"] = str(exc)
        result["summary"] = f"Syntax error: {exc}"
        return result

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as fh:
            fh.write(code)
            tmp_path = fh.name

        proc = subprocess.run(
            ["python", tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        result["execution"] = {
            "returncode": proc.returncode,
            "stdout": proc.stdout[:2000],
            "stderr": proc.stderr[:2000],
        }
        Path(tmp_path).unlink(missing_ok=True)
    except subprocess.TimeoutExpired:
        result["execution"] = {"error": "Timed out (30 s limit)"}
    except Exception as exc:
        result["execution"] = {"error": str(exc)}

    ok = result.get("execution", {}).get("returncode") == 0
    result["summary"] = ("Code validated successfully" if ok
                          else "Code validation failed")
    return result


@_register("index_document")
def _handle_index_document(text: str, metadata: dict | None = None) -> dict:
    doc_id = _get_rag().add_document(text, metadata=metadata or {})
    return {
        "doc_id": doc_id,
        "indexed_chars": len(text),
        "summary": f"Indexed {len(text)} chars into knowledge base",
    }


@_register("list_repo_structure")
def _handle_list_repo_structure(repo_path: str, max_depth: int = 3) -> dict:
    root = Path(repo_path)
    if not root.exists():
        return {"error": "Path not found",
                "summary": f"Path not found: {repo_path}"}

    entries: list[str] = []

    def _walk(p: Path, depth: int, prefix: str = ""):
        if depth > max_depth:
            return
        try:
            children = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        except PermissionError:
            return
        for child in children:
            if child.name.startswith("."):
                continue
            entries.append(f"{prefix}{child.name}{'/' if child.is_dir() else ''}")
            if child.is_dir():
                _walk(child, depth + 1, prefix + "  ")

    _walk(root, 0)
    return {
        "structure": entries[:300],
        "total_entries": len(entries),
        "summary": f"Listed {len(entries)} entries in {repo_path}",
    }

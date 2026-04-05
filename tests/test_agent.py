"""Smoke tests for the Paper2Agent API and components."""

import pytest

from app import create_app
from app.parsers.paper_parser import PaperParser
from app.retrieval.rag import RAGRetriever


@pytest.fixture
def client():
    app = create_app({"TESTING": True})
    with app.test_client() as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "healthy"


def test_index_and_search(client):
    resp = client.post("/api/index", json={
        "text": "Transformers use self-attention to model long-range dependencies.",
        "metadata": {"source": "test"},
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert "doc_id" in data

    resp = client.post("/api/search", json={"query": "attention mechanism", "top_k": 3})
    assert resp.status_code == 200
    results = resp.get_json()["results"]
    assert len(results) >= 1


def test_list_workflows(client):
    resp = client.get("/api/workflows")
    assert resp.status_code == 200
    assert "workflows" in resp.get_json()


def test_convert_missing_source(client):
    resp = client.post("/api/convert", json={"query": "test"})
    assert resp.status_code == 400


def test_parser_raw_text():
    parser = PaperParser()
    result = parser.parse("This is a test abstract about neural networks.")
    assert "title" in result
    assert "sections" in result


def test_parser_github():
    parser = PaperParser()
    result = parser.parse("https://github.com/The-Christopher-Robin/autotriage-distributed")
    assert result.get("title") is not None


def test_rag_add_and_search():
    rag = RAGRetriever(index_path="data/test_index")
    rag.add_document(
        "Convolutional neural networks are widely used for image classification.",
        metadata={"section": "intro"},
    )
    results = rag.search("image classification with CNNs", top_k=2)
    assert len(results) >= 1
    assert results[0]["score"] > 0

    import shutil
    shutil.rmtree("data/test_index", ignore_errors=True)

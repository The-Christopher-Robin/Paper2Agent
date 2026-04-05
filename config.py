import os


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
    WORKFLOW_DIR = os.getenv("WORKFLOW_DIR", "data/workflows")
    MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "10"))
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    DATABASE_URL = os.getenv(
        "DATABASE_URL", "sqlite:///paper2agent.db"
    )

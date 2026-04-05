"""Database session management with SQLAlchemy.

Provides engine creation, session factory, and a context-manager helper
for request-scoped database access.  Falls back to SQLite when no
DATABASE_URL is configured (local development).
"""

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import Config

logger = logging.getLogger(__name__)

_db_url = Config.DATABASE_URL
if _db_url.startswith("sqlite"):
    engine = create_engine(_db_url, connect_args={"check_same_thread": False})
else:
    engine = create_engine(_db_url, pool_pre_ping=True, pool_size=5, max_overflow=10)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()


@contextmanager
def get_db():
    """Yield a transactional database session that auto-closes."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Create all tables that don't yet exist."""
    from app import models as _models  # noqa: F401 — register models
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured (url=%s)", _db_url.split("@")[-1] if "@" in _db_url else _db_url)

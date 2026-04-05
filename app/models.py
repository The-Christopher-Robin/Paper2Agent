"""SQLAlchemy models for workflow persistence, agent traces, and documents."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.types import TypeDecorator, TEXT
from sqlalchemy.orm import relationship
import json

from app.db import Base


class JSONType(TypeDecorator):
    """Platform-independent JSON column (uses TEXT storage on SQLite)."""
    impl = TEXT
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return None


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Workflow(Base):
    __tablename__ = "workflows"

    id = Column(String(36), primary_key=True, default=_uuid)
    source_type = Column(String(50), nullable=False, default="paper")
    source_ref = Column(Text, nullable=False, default="")
    summary = Column(Text, nullable=True)
    recommendations = Column(JSONType, nullable=True, default=list)
    status = Column(String(30), nullable=False, default="in_progress")
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    steps = relationship("WorkflowStep", back_populates="workflow", cascade="all, delete-orphan")
    traces = relationship("AgentTrace", back_populates="workflow", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "summary": self.summary,
            "recommendations": self.recommendations or [],
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "steps": [s.to_dict() for s in self.steps],
        }


class WorkflowStep(Base):
    __tablename__ = "workflow_steps"

    id = Column(String(36), primary_key=True, default=_uuid)
    workflow_id = Column(String(36), ForeignKey("workflows.id"), nullable=False)
    agent_role = Column(String(60), nullable=False)
    input_text = Column(Text, nullable=True)
    output_text = Column(Text, nullable=True)
    tool_calls = Column(JSONType, nullable=True, default=list)
    step_order = Column(Integer, nullable=False, default=0)
    duration_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    workflow = relationship("Workflow", back_populates="steps")
    traces = relationship("AgentTrace", back_populates="step", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "agent_role": self.agent_role,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "tool_calls": self.tool_calls or [],
            "step_order": self.step_order,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AgentTrace(Base):
    __tablename__ = "agent_traces"

    id = Column(String(36), primary_key=True, default=_uuid)
    workflow_id = Column(String(36), ForeignKey("workflows.id"), nullable=False)
    step_id = Column(String(36), ForeignKey("workflow_steps.id"), nullable=True)
    agent_role = Column(String(60), nullable=False)
    event_type = Column(String(30), nullable=False)  # tool_call, llm_response, human_review, error
    payload = Column(JSONType, nullable=True, default=dict)
    timestamp = Column(DateTime, default=_utcnow)

    workflow = relationship("Workflow", back_populates="traces")
    step = relationship("WorkflowStep", back_populates="traces")

    def to_dict(self):
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "step_id": self.step_id,
            "agent_role": self.agent_role,
            "event_type": self.event_type,
            "payload": self.payload or {},
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=_uuid)
    content = Column(Text, nullable=False)
    embedding_text = Column(Text, nullable=True)
    source = Column(String(255), nullable=True)
    chunk_index = Column(Integer, nullable=True, default=0)
    metadata_ = Column("metadata", JSONType, nullable=True, default=dict)
    embedding = Column(Text, nullable=True)  # serialized vector; pgvector column added via migration when available
    created_at = Column(DateTime, default=_utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata_ or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

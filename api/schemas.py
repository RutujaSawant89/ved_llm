"""TrustLM API — Pydantic response/request models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceInfo(BaseModel):
    document_name: str
    page_number: Optional[int] = None
    chunk_index: int = 0
    relevance: float = 0.0
    text: str = ""
    module: str = ""
    topic: str = ""


class TrustInfo(BaseModel):
    score: str = "UNKNOWN"
    score_value: float = 0.0
    evidence_coverage: str = "UNKNOWN"
    reason: str = ""
    relevance_score: float = 0.0
    factual_support: float = 0.0
    consistency_score: float = 0.0
    chart_data: List[Dict[str, Any]] = []


class VerificationInfo(BaseModel):
    enabled: bool = True
    reason: str = ""
    supported: int = 0
    partially_supported: int = 0
    unsupported: int = 0
    grounding_rate: float = 1.0
    claims: List[Dict[str, Any]] = []


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo] = []
    trust: TrustInfo = Field(default_factory=TrustInfo)
    verification: VerificationInfo = Field(default_factory=VerificationInfo)
    generator: str = ""
    generation_available: bool = True


class DocumentUploadResponse(BaseModel):
    document_id: str
    document_name: str
    num_chunks: int
    source: str = ""


class DocumentDeleteResponse(BaseModel):
    deleted: int


class StatsResponse(BaseModel):
    total_chunks: int
    collection_name: str
    db_path: str
    generator: str


class ModelInfoResponse(BaseModel):
    architecture: Dict[str, Any] = {}


class CheckpointStatus(BaseModel):
    exists: bool = False
    steps: Optional[int] = None
    usable: bool = False
    path: str = ""
    min_steps: int = 1000


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str = "TrustLM"
    device: str = "cpu"


class TrainRequest(BaseModel):
    mode: str = "smoke_test"  # "smoke_test" or "full"
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 3e-4
    steps: Optional[int] = None


class TrainStatusResponse(BaseModel):
    running: bool = False
    step: int = 0
    max_steps: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    perplexity: float = 0.0
    tokens_processed: int = 0
    logs: List[str] = Field(default_factory=list)

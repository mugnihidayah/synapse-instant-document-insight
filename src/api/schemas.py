"""Pydantic schemas for API request/response validation"""

from datetime import datetime

from pydantic import BaseModel, Field


# SESSION SCHEMAS
class SessionCreate(BaseModel):
    """Response after creating a session"""

    session_id: str
    message: str = "Session created successfully"


class SessionInfo(BaseModel):
    """Session information"""

    session_id: str
    created_at: datetime
    document_count: int
    is_ready: bool
    ingestion_status: str = "idle"
    ingestion_error: str | None = None
    ingestion_started_at: datetime | None = None
    ingestion_completed_at: datetime | None = None


# DOCUMENT SCHEMAS
class DocumentUploadResponse(BaseModel):
    """Response after uploading documents"""

    session_id: str
    document_processed: int = 0
    chunks_created: int = 0
    files_queued: int = 0
    ingestion_status: str = "queued"
    message: str = "Documents accepted for processing"


class DocumentInfo(BaseModel):
    """Information about a processed document"""

    filename: str
    chunks: int
    status: str


class SessionDocumentItem(BaseModel):
    """Document chunk listing item for session browsing."""

    chunk_id: str
    document_id: str | None = None
    source: str | None = None
    page: int | None = None
    section: str | None = None
    chunk_type: str | None = None
    preview: str


class SessionDocumentsResponse(BaseModel):
    """Paginated chunk listing for a session."""

    session_id: str
    total: int
    page: int
    page_size: int
    items: list[SessionDocumentItem]


class QueryFilters(BaseModel):
    """Metadata filters applied before retrieval."""

    sources: list[str] | None = Field(default=None, description="Filter by source filename")
    source_type: str | None = Field(
        default=None, description="Filter by source type (pdf, txt, docx)"
    )
    page_from: int | None = Field(default=None, ge=1, description="Start page (1-indexed)")
    page_to: int | None = Field(default=None, ge=1, description="End page (1-indexed)")
    chunk_types: list[str] | None = Field(default=None, description="Filter by chunk_type metadata")
    content_origin: str | None = Field(
        default=None,
        description="Filter by content origin (text, ocr, text+ocr, table)",
    )


# QUERY SCHEMAS
class QueryRequest(BaseModel):
    """Request for RAG query"""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question to ask about the documents",
    )

    language: str = Field(
        default="id", pattern="^(id|en)$", description="Response language: 'id' or 'en'"
    )

    model: str | None = Field(default=None, description="LLM model to use (optional)")

    temperature: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Temperature for LLM (optional)",
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Retrieved document count before reranking",
    )

    rerank_top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Final number of chunks after reranking/diversification",
    )

    filters: QueryFilters | None = Field(default=None, description="Optional metadata filters")

    include_debug: bool = Field(
        default=False,
        description="Include retrieval diagnostics in response",
    )

    strict_grounding: bool = Field(
        default=True,
        description="Return fallback answer when grounding score is too low",
    )

    enable_query_rewrite: bool = Field(
        default=True,
        description="Enable query rewrite before retrieval",
    )


class SourceItem(BaseModel):
    """Individual source/citation item"""

    text: str
    snippet: str | None = None
    score: float = Field(ge=0, le=1, description="Relevance score (0-1)")
    chunk_id: str = Field(description="Unique chunk identifier")
    document_id: str | None = Field(default=None, description="Original uploaded document id")
    source: str | None = None
    page: int | None = None
    metadata: dict = Field(default_factory=dict)


class QueryDebug(BaseModel):
    """Retrieval debug payload."""

    rewritten_query: str
    retrieved_count: int
    reranked_count: int
    top_k_used: int
    rerank_top_k_used: int
    filters_applied: dict | None = None


class QueryResponse(BaseModel):
    """Response from RAG query"""

    answer: str
    sources: list[SourceItem]
    model_used: str
    rewritten_query: str | None = None
    grounded: bool = True
    grounding_score: float = Field(default=1.0, ge=0, le=1)
    debug: QueryDebug | None = None


class FeedbackRequest(BaseModel):
    """User feedback payload."""

    question: str = Field(..., min_length=1, max_length=5000)
    answer: str = Field(..., min_length=1, max_length=20000)
    rating: int = Field(..., ge=-1, le=1, description="-1 negative, 0 neutral, 1 positive")
    comment: str | None = Field(default=None, max_length=2000)
    metadata: dict = Field(default_factory=dict)


class FeedbackResponse(BaseModel):
    """Persisted feedback response."""

    feedback_id: str
    session_id: str
    rating: int
    created_at: datetime


class QuotaInfo(BaseModel):
    """Daily query quota stats."""

    daily_limit: int
    used_today: int
    remaining_today: int


class UsageResponse(BaseModel):
    """Usage and analytics summary."""

    key_id: str
    total_sessions: int
    total_queries: int
    total_documents: int
    total_feedback: int
    quota: QuotaInfo


# ERROR SCHEMAS
class ErrorResponse(BaseModel):
    """Error response"""

    error: str
    detail: str | None = None

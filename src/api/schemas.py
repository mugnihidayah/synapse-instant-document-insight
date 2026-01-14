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

# DOCUMENT SCHEMAS
class DocumentUploadResponse(BaseModel):
  """Response after uploading documents"""

  session_id: str
  document_processed: int
  chunks_created: int
  message: str = "Documents processed successfully"

class DocumentInfo(BaseModel):
  """Information about a processed document"""

  filename: str
  chunks: int
  status: str

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
    default="id",
    pattern="^(id|en)$",
    description="Response language: 'id' or 'en'"
  )

  model:str | None = Field(
    default=None,
    description="LLM model to use (optional)"
  )

class QueryResponse(BaseModel):
  """Response from RAG query"""

  answer: str
  sources: list[dict]
  model_used: str

# ERROR SCHEMAS
class ErrorResponse(BaseModel):
  """Error response"""

  error: str
  detail: str | None = None
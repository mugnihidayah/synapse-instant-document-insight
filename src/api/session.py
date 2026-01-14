"""
Session management for RAG

Manages vectorstores per session to support multiple concurrent users
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class Session:
  """Represents a user session with its vectorstore"""

  id: str
  created_at: datetime = field(default_factory=datetime.now)
  vectorstore: Any = None
  document_count: int = 0

  def is_ready(self) -> bool:
    """Check if session has documents loaded"""
    return self.vectorstore is not None and self.document_count > 0

class SessionManager:
  """
  Manages user sessions and their vectorstores

  In production, this would be backed by a database
  For now, it uses an in-memory storage
  """

  def __init__(self):
    self._sessions: dict[str, Session] = {}

  def create_session(self) -> Session:
    """Create a new session"""
    session_id = uuid.uuid4().hex[:12]
    session = Session(id=session_id)
    self._sessions[session_id] = session
    return session

  def get_session(self, session_id: str) -> Session | None:
    """Get a session by ID"""
    return self._sessions.get(session_id)

  def update_session(self, session_id:str, vectorstore: Any, document_count: int) -> Session | None:
    """Update session with vectorstore"""
    session = self._sessions.get(session_id)
    if session:
      session.vectorstore = vectorstore
      session.document_count = document_count
    return session

  def delete_session(self, session_id: str) -> bool:
    """Delete a session"""
    if session_id in self._sessions:
      del self._sessions[session_id]
      return True
    return False

  def list_sessions(self) -> list[Session]:
    """List all sessions"""
    return list(self._sessions.values())

# global session manager instance
session_manager = SessionManager()
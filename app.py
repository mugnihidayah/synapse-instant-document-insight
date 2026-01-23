import streamlit as st
import httpx
import os
from io import BytesIO

# API Configuration
API_BASE_URL = os.getenv(
  "API_BASE_URL", 
  "https://synapse-instant-document-insight-production.up.railway.app"
)

# SESSION STATE
if "messages" not in st.session_state:
  st.session_state.messages = []

if "uploader_key" not in st.session_state:
  st.session_state.uploader_key = 0

if "session_id" not in st.session_state:
  st.session_state.session_id = None

if "api_key" not in st.session_state:
  st.session_state.api_key = None

if "active_document" not in st.session_state:
  st.session_state.active_document = None

# PAGE CONFIGURATION
st.set_page_config(
  page_title="Synapse",
  page_icon="🧠",
  layout="wide",
)

# HELPER FUNCTIONS
def get_api_key():
  """Get or create API key"""
  if st.session_state.api_key:
    return st.session_state.api_key
  
  try:
    response = httpx.post(
      f"{API_BASE_URL}/api/v1/keys/",
      json={"name": "streamlit-app"},
      timeout=30.0
    )
    if response.status_code == 200:
      data = response.json()
      st.session_state.api_key = data["api_key"]
      return st.session_state.api_key
  except Exception as e:
    st.error(f"Failed to get API key: {e}")
  return None


def create_session(api_key: str):
  """Create a new session"""
  try:
    response = httpx.post(
      f"{API_BASE_URL}/api/v1/documents/sessions",
      headers={"X-API-Key": api_key},
      timeout=30.0
    )
    if response.status_code == 200:
      data = response.json()
      return data["session_id"]
  except Exception as e:
    st.error(f"Failed to create session: {e}")
  return None


def upload_files(api_key: str, session_id: str, files):
  """Upload files to session"""
  try:
    # Prepare files for upload
    file_data = []
    for f in files:
      file_data.append(("files", (f.name, f.getvalue(), f.type or "application/octet-stream")))
    
    response = httpx.post(
      f"{API_BASE_URL}/api/v1/documents/upload/{session_id}",
      headers={"X-API-Key": api_key},
      files=file_data,
      timeout=120.0  # Longer timeout for file uploads
    )
    if response.status_code == 200:
      return response.json()
  except Exception as e:
    st.error(f"Failed to upload files: {e}")
  return None


def query_documents(api_key: str, session_id: str, question: str, language: str = "id", temperature: float = 0.3):
  """Query documents"""
  try:
    response = httpx.post(
      f"{API_BASE_URL}/api/v1/query/{session_id}",
      headers={"X-API-Key": api_key},
      json={
        "question": question,
        "language": language,
        "temperature": temperature
      },
      timeout=60.0
    )
    if response.status_code == 200:
      return response.json()
  except Exception as e:
    st.error(f"Failed to query: {e}")
  return None


def delete_session(api_key: str, session_id: str):
  """Delete a session"""
  try:
    response = httpx.delete(
      f"{API_BASE_URL}/api/v1/documents/sessions/{session_id}",
      headers={"X-API-Key": api_key},
      timeout=30.0
    )
    return response.status_code == 200
  except Exception:
    return False


# HEADER
st.title("🧠 Synapse: Instant Document Insights")
st.caption("🚀 Advanced RAG System - Powered by FastAPI Backend")

# SIDEBAR (CONTROL)
with st.sidebar:
  st.header("🎛️ Control Panel")

  # Initialize API Key
  if not st.session_state.api_key:
    with st.spinner("Connecting to API..."):
      get_api_key()
  
  if st.session_state.api_key:
    st.success("✅ Connected to API")
  else:
    st.error("❌ API connection failed")

  if st.button("💬 New Chat", type="primary"):
    # Delete old session if exists
    if st.session_state.session_id and st.session_state.api_key:
      delete_session(st.session_state.api_key, st.session_state.session_id)
    st.session_state.messages = []
    st.session_state.session_id = None
    st.session_state.active_document = None
    st.success("Chat history cleared! Start a new conversation.")
    st.rerun()

  st.divider()

  st.subheader("📄 Active Document")
  if st.session_state.active_document:
    st.success(f"✅ {st.session_state.active_document['name']}")
    st.caption(f"📊 {st.session_state.active_document['chunks']} chunks processed")
    st.caption(f"🔑 Session: {st.session_state.session_id[:8]}...")
  else:
    st.warning("No document loaded")
  
  st.divider()

  # Settings
  temperature = st.slider("Creativity (Temperature):", 0.0, 1.0, 0.3)

  st.divider()

  # Upload Documents
  st.subheader("📂 Upload Documents")
  uploaded_files = st.file_uploader(
    "Upload PDF/TXT/DOCX", 
    accept_multiple_files=True,
    type=["pdf", "txt", "docx"],
    key=f"uploader_{st.session_state.uploader_key}"
  )
  
  if st.button("📤 Process Documents"):
    if not st.session_state.api_key:
      st.error("API not connected")
    elif not uploaded_files:
      st.warning("Please select files first")
    else:
      with st.spinner("Processing documents..."):
        # Create new session
        session_id = create_session(st.session_state.api_key)
        if session_id:
          st.session_state.session_id = session_id
          
          # Upload files
          result = upload_files(st.session_state.api_key, session_id, uploaded_files)
          if result:
            st.session_state.active_document = {
              "name": ", ".join([f.name for f in uploaded_files]),
              "chunks": result.get("chunks_created", len(uploaded_files)),
            }
            st.session_state.messages = []
            st.success(f"✅ Processed {len(uploaded_files)} documents!")
            st.session_state.uploader_key += 1
            st.rerun()
          else:
            st.error("Failed to upload files")
        else:
          st.error("Failed to create session")

  # Reset
  st.divider()
  if st.button("🗑️ Reset Session", type="secondary"):
    if st.session_state.session_id and st.session_state.api_key:
      delete_session(st.session_state.api_key, st.session_state.session_id)
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.active_document = None
    st.success("Session cleared!")
    st.rerun()

  # Language Selector
  language = st.selectbox(
    "Language Answer:",
    ("Indonesia", "English"),
    index=0,
  )
  lang_code = "id" if language == "Indonesia" else "en"

  st.divider()
  
  # API Status
  st.caption(f"🌐 API: {API_BASE_URL}")

# RENDER CHAT HISTORY
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# INPUT USER
if prompt := st.chat_input("Ask a question about your document..."):
  if not st.session_state.session_id:
    st.error("Please upload a document first!")
  elif not st.session_state.api_key:
    st.error("API not connected")
  else:
    # Display user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)
    
    # Get answer from API
    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        result = query_documents(
          st.session_state.api_key,
          st.session_state.session_id,
          prompt,
          language=lang_code,
          temperature=temperature
        )
        
        if result:
          answer = result.get("answer", "No answer received")
          sources = result.get("sources", [])
          
          st.markdown(answer)
          
          # Show sources
          if sources:
            with st.expander("📚 Reference Sources"):
              for i, src in enumerate(sources):
                metadata = src.get("metadata", {})
                text = src.get("text", "No text available")
                st.markdown(f"**Source {i+1}:** {metadata.get('source', 'Unknown')}")
                with st.expander(f"📄 View quoted text from Source {i+1}"):
                  st.info(text[:500] + "..." if len(text) > 500 else text)
          
          st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
          st.error("Failed to get answer from API")
import streamlit as st
import gc
import time
from src.ingestion.loaders import load_documents_from_uploads
from src.ingestion.chunkers import split_documents
from src.ingestion.vectorstore import create_vectorstore
from src.rag.chain import ask_question
from src.core.config import settings

# SESSION STATE
if "messages" not in st.session_state:
  st.session_state.messages = []

if "uploader_key" not in st.session_state:
  st.session_state.uploader_key = 0

if "vectorstore" not in st.session_state:
  st.session_state.vectorstore = None

if "active_document" not in st.session_state:
  st.session_state.active_document = None

# PAGE CONFIGURATION
st.set_page_config(
  page_title="Synapse",
  page_icon="🧠",
  layout="wide",
)

# HEADER
st.title("🧠 Synapse: Instant Document Insights")
st.caption("🚀 Advanced RAG System")

# SIDEBAR (CONTROL)
with st.sidebar:
  st.header("🎛️ Control Panel")

  if st.button("💬 New Chat", type="primary"):
    st.session_state.messages = []
    st.success("Chat history cleared! Start a new conversation.")
    st.rerun()

  st.divider()

  st.subheader("📄 Active Document")
  if st.session_state.active_document:
    st.success(f"✅ {st.session_state.active_document['name']}")
    st.caption(f"📊 {st.session_state.active_document['chunks']} chunks processed")
  else:
    st.warning("No document loaded")
  
  st.divider()

  # Model Selector
  selected_model = st.selectbox("Select AI Model:", settings.available_llms, index=0)
  temperature = st.slider("Creativity (Temperature):", 0.0, 1.0, 0.3)

  st.divider()

  # Upload Documents
  st.subheader("📂 Upload Documents")
  st.markdown("It is recommended to reset the database before uploading documents.")
  uploaded_files = st.file_uploader(
    "Upload PDF/TXT/DOCX", 
    accept_multiple_files=True,
    type=["pdf", "txt", "docx"],
    key=f"uploader_{st.session_state.uploader_key}"
  )
  
  if st.button("Document Process (Ingest)"):
    with st.spinner("Processing documents..."):
      if uploaded_files:
        if st.session_state.vectorstore is not None:
          del st.session_state.vectorstore
        st.session_state.vectorstore = None
        st.session_state.messages = []
        gc.collect()
        # Process directly from memory, not saving to data/
        docs = load_documents_from_uploads(uploaded_files)
        chunks = split_documents(docs)
        st.session_state.vectorstore = create_vectorstore(chunks)
        st.session_state.active_document = {
          "name": ", ".join([f.name for f in uploaded_files]),
          "chunks": len(chunks),
        }
        st.success(f"Successfully processed {len(uploaded_files)} documents!")
        st.session_state.uploader_key += 1
        time.sleep(3)
        st.rerun()
      else:
        st.warning("Select the file before processing.")

  # Reset Database
  st.divider()
  if st.button("🗑️ Reset Database", type="secondary"):
    st.session_state.vectorstore = None
    st.session_state.messages = []
    st.session_state.active_document = None
    st.success("Database & chat cleared!")
    time.sleep(1)
    st.rerun()

  # Language Selector
  language = st.selectbox(
    "Language Answer:",
    ("Indonesia", "English"),
    index=0,
  )

  lang_code = "id" if language == "Indonesia" else "en"

  st.divider()

# RENDER CHAT HISTORY (Supaya chat lama tetap ada)
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])
# INPUT USER
if prompt := st.chat_input("Ask a question about this document..."):
  # Display user questions
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)
  # Answer Process
  with st.chat_message("assistant"):
    try:
      # Call our RAG function
      stream_generator, sources = ask_question(
        prompt, 
        st.session_state.messages,
        st.session_state.vectorstore,
        model_name=selected_model, 
        temperature=temperature,
        language=lang_code,
      )
      
      # Render Streaming Response
      response = st.write_stream(stream_generator)
      
      # Show Source (Collapsible)
      if sources:
        with st.expander("📚 Reference Sources"):
          for i, src in enumerate(sources):
            metadata = src.get('metadata', {})
            text = src.get('text', 'No text available')
            
            st.markdown(f"**Source {i+1}:** {metadata.get('source', 'Unknown')} (Page: {metadata.get('page', '-')})")
            
            with st.expander(f"📄 View quoted text from Source {i+1}"):
              st.info(text)
      # Save assistant's answer to history
      st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
      st.error(f"An error occurred: {e}")
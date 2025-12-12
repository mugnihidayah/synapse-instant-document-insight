import streamlit as st
import os
import shutil
import gc
import time
from ingest import load_documents_from_files, split_documents, create_vectorstore
from chain import ask_question
from config import AVAILABLE_LLMS

# SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

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

  # Model Selector
  selected_model = st.selectbox("Select AI Model:", AVAILABLE_LLMS, index=0)
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
        # Process directly from memory, not saving to data/
        docs = load_documents_from_files(uploaded_files)
        chunks = split_documents(docs)
        st.session_state.vectorstore = create_vectorstore(chunks)
        
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
                        st.markdown(f"**Source {i+1}:** {src.get('source', 'Unknown')} (Page: {src.get('page', '-')})")
            # Save assistant's answer to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
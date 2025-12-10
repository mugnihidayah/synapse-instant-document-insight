import streamlit as st
import os
import shutil
import gc
from ingest import load_documents, split_documents, add_to_chroma, clear_database
from chain import ask_question
from config import DATA_PATH, AVAILABLE_LLMS

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
      type=["pdf", "txt", "docx"]
  )
  
  if st.button("Document Process (Ingest)"):
      with st.spinner("Processing documents..."):
          # Create a data folder if it does not already exist.
          if not os.path.exists(DATA_PATH):
              os.makedirs(DATA_PATH)
          
          # Save uploaded files to the data folder
          if uploaded_files:
              for old_file in os.listdir(DATA_PATH):
                old_file_path = os.path.join(DATA_PATH, old_file)
                try:
                    os.remove(old_file_path)
                except:
                    pass
              for uploaded_file in uploaded_files:
                  save_path = os.path.join(DATA_PATH, uploaded_file.name)
                  with open(save_path, "wb") as f:
                      f.write(uploaded_file.getbuffer())
              
              clear_database()
              # Run the ingest process
              docs = load_documents()
              chunks = split_documents(docs)
              add_to_chroma(chunks)
              st.cache_resource.clear()
              st.success(f"Successfully processed {len(uploaded_files)} documents!")
              st.rerun() 
          else:
              st.warning("Select the file before processing.")

  # Reset Database
  st.divider()
  if st.button("🗑️ Reset Database", type="secondary"):
      st.cache_resource.clear()
      gc.collect()
      
      with st.spinner("Deleting data..."):
          if clear_database():
              # Delete physical files in the data folder
              if os.path.exists(DATA_PATH):
                  for filename in os.listdir(DATA_PATH):
                      file_path = os.path.join(DATA_PATH, filename)
                      try:
                          os.remove(file_path)
                      except Exception as e:
                          print(f"Failed to delete {filename}: {e}")
              
              st.success("Database & uploaded files have been cleared!")
              st.session_state.messages = []
              st.rerun()
          else:
              st.error("Failed to delete database. Try restarting the application.")

  # Language Selector
  language = st.selectbox(
    "Language Answer:",
    ("Indonesia", "English"),
    index=0,
  )

  lang_code = "id" if language == "Indonesia" else "en"

  st.divider()

# SESSION STATE (Chat History)
if "messages" not in st.session_state:
    st.session_state.messages = []
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
import streamlit as st
import os
import time
import uuid
from pdf_handler import (
    extract_text_from_pdfs,
    chunk_text,
    load_model,
    load_local_llm,
    embed_texts,
    build_index,
    semantic_search,
    generate_answer
)

# Page Config
st.set_page_config(page_title="QueryIQ", page_icon="📚", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stChatInput {
        border-radius: 20px;
    }
    div[data-testid="stCard"] {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def save_uploaded_files(uploaded_files):
    """Saves uploaded files to disk temporarily."""
    saved_paths = []
    for uploaded_file in uploaded_files:
        try:
            path = uploaded_file.name
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(path)
        except Exception as e:
            st.error(f"Error saving file {uploaded_file.name}: {e}")
    return saved_paths

def init_session_state():
    """Initialize session state variables."""
    if "chats" not in st.session_state:
        st.session_state.chats = {}  # {chat_id: {"title": "...", "messages": [...]}}
    
    if "current_chat_id" not in st.session_state:
        new_chat_id = str(uuid.uuid4())
        st.session_state.chats[new_chat_id] = {"title": "New Chat", "messages": []}
        st.session_state.current_chat_id = new_chat_id

    if "local_llm" not in st.session_state:
        st.session_state.local_llm = None

def main():
    init_session_state()
    
    st.title("📚 QueryIQ: Advanced RAG Assistant")

    # Sidebar
    with st.sidebar:
        st.header("🤖 AI Configuration")
        
        # Model Selection
        model_option = st.radio(
            "Model Power",
            ["High (Cloud/Gemini)", "Low (Local/Offline)"],
            index=0,
            help="High: Uses Google Gemini (Requires API Key). Low: Uses local LaMini model (Offline, less accurate)."
        )
        
        api_key = None
        if "High" in model_option:
            api_key = st.text_input("Google Gemini API Key", type="password", help="Get your key from AI Studio")
            if api_key:
                st.session_state.api_key = api_key
        else:
            # Lazy load local model
            if st.session_state.local_llm is None:
                with st.spinner("⚙️ Loading Local AI Model (LaMini)... This happens once."):
                    st.session_state.local_llm = load_local_llm()
            st.info("Using Local AI (LaMini-Flan-T5). No internet required.")

        st.divider()
        
        # Chat Management
        st.header("💬 Chats")
        if st.button("➕ New Chat", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
            st.session_state.current_chat_id = new_id
            st.rerun()
            
        # Chat List
        chat_ids = list(st.session_state.chats.keys())
        if chat_ids:
            # Reverse list to show newest first
            chat_ids.reverse()
            
            # Find index of current chat in reversed list
            try:
                current_index = chat_ids.index(st.session_state.current_chat_id)
            except ValueError:
                current_index = 0
                
            selected_chat_id = st.selectbox(
                "Switch Chat",
                options=chat_ids,
                format_func=lambda x: st.session_state.chats[x]["title"],
                index=current_index
            )
            
            if selected_chat_id != st.session_state.current_chat_id:
                st.session_state.current_chat_id = selected_chat_id
                st.rerun()

        st.divider()
        
        st.header("📂 Document Upload")
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type="pdf", 
            accept_multiple_files=True
        )
        
        if st.button("Process Documents") and uploaded_files:
            with st.spinner("Processing PDFs..."):
                pdf_paths = save_uploaded_files(uploaded_files)
                documents = extract_text_from_pdfs(pdf_paths)
                chunks = chunk_text(documents, chunk_size=1000, overlap=100)
                
                model = load_model()
                embeddings = embed_texts(chunks, model)
                index = build_index(embeddings)
                
                st.session_state.chunks = chunks
                st.session_state.model = model
                st.session_state.index = index
                st.session_state.processed = True
                
                st.success(f"✅ Processed {len(uploaded_files)} files!")
                
                for path in pdf_paths:
                    if os.path.exists(path):
                        os.remove(path)

    # Chat Interface
    current_chat_id = st.session_state.current_chat_id
    # Ensure current chat exists (persistence check)
    if current_chat_id not in st.session_state.chats:
        st.session_state.chats[current_chat_id] = {"title": "New Chat", "messages": []}
        
    current_chat = st.session_state.chats[current_chat_id]
    
    # Display History
    for message in current_chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input
    if prompt := st.chat_input("Ask a question..."):
        # Update Chat Title if it's the first message
        if len(current_chat["messages"]) == 0:
            current_chat["title"] = prompt[:30] + "..."
            
        # User Message
        current_chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI Response
        if "index" in st.session_state:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # 1. Retrieve
                    results = semantic_search(
                        prompt,
                        st.session_state.model,
                        st.session_state.index,
                        st.session_state.chunks,
                        top_k=5
                    )
                    
                    # 2. Generate
                    model_choice = "high" if "High" in model_option else "low"
                    
                    answer = generate_answer(
                        prompt, 
                        results, 
                        model_type=model_choice,
                        api_key=st.session_state.get("api_key"),
                        local_llm=st.session_state.get("local_llm")
                    )
                    
                    st.markdown(answer)
                    
                    # Sources
                    with st.expander("View Sources"):
                        for i, res in enumerate(results):
                            st.markdown(f"**Source {i+1}** ({res['chunk']['source']}, p.{res['chunk']['page']})")
                            st.caption(res['chunk']['text'][:300] + "...")
                            st.divider()
                    
                    current_chat["messages"].append({"role": "assistant", "content": answer})
        else:
            with st.chat_message("assistant"):
                st.warning("Please upload and process documents first!")

if __name__ == "__main__":
    main()

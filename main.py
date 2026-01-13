# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import os
import time
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
import pickle
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(layout="wide")

# Check API key


# ============================================
# OPTIMIZATION 1: Model Selection for Low Latency
# ============================================
with st.sidebar:
    st.subheader("⚡ Model Configuration (Low Latency)")
    
    # Let user choose, but default to Mistral 7B for latency
    model_choice = st.selectbox(
        "Choose Model (Mistral 7B recommended for speed):",
        [
            "mistralai/mistral-7b-instruct-v0.3",  # Best for latency
            "meta/llama3-8b-instruct",             # Better accuracy
            "google/codegemma-7b",                 # Alternative
        ],
        index=0  # Default to Mistral
    )
    
    # Embedding model optimized for speed
    embed_model = st.selectbox(
        "Embedding Model:",
        [
            "NV-Embed-QA",  # Fast and good for QA
            "nvidia/nv-embedqa-e5",  # Alternative
        ],
        index=0
    )

# Initialize models with caching for performance
@st.cache_resource
def initialize_models(llm_model, embed_model):
    """Cache models to avoid re-initialization"""
    try:
        llm = ChatNVIDIA(
            model=llm_model,
            temperature=0.1,  # Lower temperature for faster, more consistent responses
            max_tokens=512,   # Limit response length for speed
        )
        document_embedder = NVIDIAEmbeddings(model=embed_model, model_type="passage")
        return llm, document_embedder
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return None, None

llm, document_embedder = initialize_models(model_choice, embed_model)

if not llm:
    st.stop()

# ============================================
# OPTIMIZATION 2: Faster Document Processing
# ============================================
with st.sidebar:
    DOCS_DIR = os.path.abspath("./uploaded_docs")
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    
    st.subheader("📁 Document Management")
    
    # Quick upload without complex UI
    uploaded_files = st.file_uploader(
        "Upload documents:", 
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx', 'md']
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DOCS_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} file(s)")

# ============================================
# OPTIMIZATION 3: Efficient Vector Store Setup
# ============================================
vector_store_path = "vectorstore.pkl"

# Use caching for document loading and processing
@st.cache_resource
def load_and_process_documents(_embedder, docs_dir, recreate=False):
    """Load and process documents with caching"""
    if not os.path.exists(docs_dir) or not os.listdir(docs_dir):
        return None
    
    # Check if we have a cached vector store
    if not recreate and os.path.exists(vector_store_path):
        try:
            with open(vector_store_path, "rb") as f:
                return pickle.load(f)
        except:
            pass
    
    # Process documents
    raw_documents = DirectoryLoader(docs_dir).load()
    if not raw_documents:
        return None
    
    # Optimized chunking for RAG
    text_splitter = CharacterTextSplitter(
        chunk_size=400,  # Smaller chunks for faster retrieval
        chunk_overlap=100,
        separator="\n",
        length_function=len,
    )
    
    documents = text_splitter.split_documents(raw_documents)
    
    # Create vector store with optimized parameters
    vectorstore = FAISS.from_documents(
        documents, 
        _embedder,
        distance_strategy="COSINE"  # Fast similarity calculation
    )
    
    # Save for future use
    with open(vector_store_path, "wb") as f:
        pickle.dump(vectorstore, f)
    
    return vectorstore

# Load vector store
vectorstore = load_and_process_documents(document_embedder, DOCS_DIR)

# ============================================
# OPTIMIZATION 4: Fast RAG Pipeline
# ============================================
st.title("⚡ Low-Latency RAG Chatbot")
st.markdown(f"**Using:** `{model_choice}` | **Embeddings:** `{embed_model}`")

# Optimized prompt template for speed
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are Envie, a fast and helpful AI assistant. Answer concisely using the context if available."),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

# Create chain with optimization
chain = prompt_template | llm | StrOutputParser()

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.latencies = []  # Track response times

# Display chat history
for msg in st.session_state.messages[-10:]:  # Limit display for speed
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Start timing
    start_time = time.time()
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # FAST RETRIEVAL (if vectorstore exists)
            context = ""
            if vectorstore:
                # Optimized retrieval with fewer documents
                retriever = vectorstore.as_retriever(
                    search_kwargs={
                        "k": 3,  # Retrieve only 3 most relevant chunks (was 4-5)
                        "score_threshold": 0.7  # Filter low relevance
                    }
                )
                docs = retriever.invoke(user_input)
                if docs:
                    context = "\n".join([doc.page_content[:300] for doc in docs])  # Limit context length
            
            # Prepare input
            chain_input = {
                "context": context if context else "No specific context available.",
                "question": user_input
            }
            
            # Stream response
            for chunk in chain.stream(chain_input):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Calculate and display latency
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            st.session_state.latencies.append(latency)
            
            # Show latency (optional)
            with st.expander("⚡ Performance Info", expanded=False):
                avg_latency = sum(st.session_state.latencies[-5:]) / min(5, len(st.session_state.latencies))
                st.caption(f"Response time: **{latency:.0f}ms** | Avg (last 5): **{avg_latency:.0f}ms**")
                st.progress(min(100, latency / 10))
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            full_response = "I apologize, but I encountered an error. Please try again."
            message_placeholder.markdown(full_response)
    
    # Add to session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Auto-scroll to latest message
    st.rerun()

# Sidebar stats
with st.sidebar:
    if st.session_state.latencies:
        avg_lat = sum(st.session_state.latencies) / len(st.session_state.latencies)
        st.metric("Average Latency", f"{avg_lat:.0f}ms")
    
    # Quick model comparison info
    with st.expander("ℹ️ Model Tips"):
        st.info("""
        **For lowest latency:**
        - Use Mistral 7B (fastest)
        - Limit responses to < 512 tokens
        - Use smaller chunks (300-400 chars)
        - Retrieve only 2-3 relevant chunks
        
        **For best accuracy:**
        - Use Llama3 8B
        - Larger chunks (500-600 chars)
        - Retrieve 4-5 chunks
        """)
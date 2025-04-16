import uuid
import time
import os
import streamlit as st
import numpy as np # Added for FAISS saving/loading potentially

# --- Constants --- (Should be moved to a config file or env vars later)
DEFAULT_CHUNK_SIZE = 500 # Tokens
DEFAULT_CHUNK_OVERLAP = 50 # Tokens
DEFAULT_MAX_CHUNKS_RETRIEVED = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.3 # Lowered threshold to allow more matches
# VECTOR_DB_PATH = "./data/chroma_db" # Removed
FAISS_INDEX_PATH = "./data/faiss_index.idx" # Added
METADATA_PATH = "./data/metadata.pkl" # Added (using pickle for simplicity)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# LLM_MODEL_NAME = "claude-3-haiku-20240307"
LLM_MODEL_NAME = "claude-3-5-sonnet-20240620" # Use Sonnet 3.5 as requested
EMBEDDING_DIM = 384 # Dimension for all-MiniLM-L6-v2

# --- ID Generation --- 
def generate_unique_id():
    """Generates a unique UUID."""
    return str(uuid.uuid4())

def generate_doc_id(filename):
    """Generates a unique ID for a document based on filename and time."""
    # Using a simple approach for now, might need refinement
    return f"{os.path.splitext(filename)[0]}_{int(time.time())}"

def generate_chunk_id(doc_id, chunk_index):
    """Generates a unique ID for a chunk within a document."""
    return f"{doc_id}_chunk_{chunk_index}"

# --- Configuration Management --- 
def load_rag_configs():
    """Loads RAG configurations. Placeholder for loading from file/db."""
    # For now, just return the default
    return {
        "default": create_default_rag_config(),
        # Add more preset configs here if needed
    }

def create_default_rag_config():
    """Creates the default RAG configuration dictionary."""
    return {
        "id": "default",
        "name": "Default",
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "max_chunks": DEFAULT_MAX_CHUNKS_RETRIEVED, # 'k' for FAISS search
        "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": LLM_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM # Added dimension info
    }

def get_config_for_conversation(convo_id):
    """Gets the RAG config for a specific conversation (or default)."""
    # Placeholder: In a real app, conversations would store their config ID
    # For now, always return the default loaded config
    configs = load_rag_configs()
    # Assume conversation has a config_id field, fetch st.session_state...convo[convo_id]['config_id']
    # return configs.get(st.session_state.conversations[convo_id].get('config_id', 'default'), configs['default'])
    return configs['default'] # Simplified for now


# --- Session State Initialization --- 
def initialize_session_state():
    """Initializes Streamlit session state variables if they don't exist."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = generate_unique_id() # Still useful potentially

    if "documents" not in st.session_state:
        st.session_state.documents = {} # {doc_id: {name, size, token_count, chunk_ids}}

    if "conversations" not in st.session_state:
        st.session_state.conversations = {} # {convo_id: {name, messages, rag_config_id}}

    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None

    if "rag_configs" not in st.session_state:
        st.session_state.rag_configs = load_rag_configs()

    if "selected_docs_for_query" not in st.session_state:
        st.session_state.selected_docs_for_query = [] # List of doc_ids or ['all']

    if "vector_store_initialized" not in st.session_state:
        # Check if index file exists as a proxy for initialization
        st.session_state.vector_store_initialized = os.path.exists(FAISS_INDEX_PATH)

    if "processing_files" not in st.session_state:
         st.session_state.processing_files = False # Flag to prevent re-processing during rerun

# --- Vector Store Naming --- (Removed Chroma specific naming)

# --- Other Helpers --- 
def get_current_timestamp():
    """Returns the current timestamp."""
    return time.time() 
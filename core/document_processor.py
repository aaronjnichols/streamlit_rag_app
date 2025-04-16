import streamlit as st
import PyPDF2
import tiktoken
import time
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

from .utils import (
    generate_doc_id,
    generate_chunk_id,
    get_current_timestamp,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM
)

# --- Initialize session state for vector store ---
def initialize_vector_store_state():
    """Initialize session state variables for vector store if they don't exist."""
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "metadata_map" not in st.session_state:
        st.session_state.metadata_map = {}
    if "chunk_id_to_faiss_id" not in st.session_state:
        st.session_state.chunk_id_to_faiss_id = {}
    if "next_faiss_id" not in st.session_state:
        st.session_state.next_faiss_id = 0

# Always ensure vector store state is initialized
initialize_vector_store_state()

# --- FAISS Index and Metadata Loading/Saving ---

def load_vector_store():
    """Initializes a NEW, empty FAISS index and metadata map for the current session."""
    # Initialize new store
    print("Initializing new in-memory FAISS index and metadata for this session.")

    # Create new index
    try:
        print(f"Creating new FAISS index (dimension: {EMBEDDING_DIM}). Using IndexIDMap2(IndexFlatIP).")
        base_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        new_faiss_index = faiss.IndexIDMap2(base_index)
        print("Successfully created new FAISS index object.")
        
        # Store in session state
        st.session_state.faiss_index = new_faiss_index
        st.session_state.metadata_map = {}
        st.session_state.chunk_id_to_faiss_id = {}
        st.session_state.next_faiss_id = 0
        st.session_state.vector_store_initialized = True
        
    except Exception as e:
        st.error(f"Failed to create new FAISS index object: {e}")
        print(f"ERROR: Failed to create new FAISS index object: {e}")
        st.session_state.faiss_index = None
        st.session_state.metadata_map = {}
        st.session_state.chunk_id_to_faiss_id = {}
        st.session_state.next_faiss_id = 0
        st.session_state.vector_store_initialized = False

    # Return references to the session state objects
    return (st.session_state.faiss_index, 
            st.session_state.metadata_map,
            st.session_state.chunk_id_to_faiss_id,
            st.session_state.next_faiss_id)


def save_vector_store():
    """Saves the FAISS index and metadata map to disk."""
    # Access from session state
    if st.session_state.faiss_index is None:
        st.warning("Attempted to save an uninitialized FAISS index. Skipping.")
        return

    try:
        print(f"Saving FAISS index ({st.session_state.faiss_index.ntotal} vectors) to {FAISS_INDEX_PATH}")
        # Ensure data directory exists
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        faiss.write_index(st.session_state.faiss_index, FAISS_INDEX_PATH)
    except Exception as e:
        st.error(f"Failed to save FAISS index: {e}")

    try:
        print(f"Saving metadata ({len(st.session_state.metadata_map)} chunks) to {METADATA_PATH}")
        os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump({
                'metadata_map': st.session_state.metadata_map,
                'chunk_id_to_faiss_id': st.session_state.chunk_id_to_faiss_id,
                'next_faiss_id': st.session_state.next_faiss_id
                }, f)
    except Exception as e:
        st.error(f"Failed to save metadata: {e}")


# --- Model/Client Initialization ---

@st.cache_resource # Cache the embedding model
def get_embedding_model():
    """Loads and caches the SentenceTransformer model."""
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # Important: Normalize embeddings for FAISS IndexFlatIP (Cosine Similarity)
    # This can sometimes be done within SentenceTransformer config or manually after encoding
    # For MiniLM, normalization is standard practice for cosine similarity search.
    print("Embedding model loaded.")
    return model

@st.cache_resource # Cache the tokenizer
def get_tokenizer():
    """Loads and caches the TikToken tokenizer."""
    print("Loading tokenizer...")
    return tiktoken.get_encoding("cl100k_base")

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file, handling potential issues."""
    text = ""
    try:
        pdf_file.seek(0) # Reset file pointer
        reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text: # Ensure text was extracted
                # Add page number identifier? Maybe less useful with semantic chunking
                # text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                text += page_text + "\n" # Add newline separator between pages
    except Exception as e:
        st.error(f"Error reading PDF {pdf_file.name}: {e}")
        return None
    return text

def chunk_text(text, chunk_size, chunk_overlap, tokenizer):
    """Splits text into chunks based on token count with overlap."""
    if not text:
        return []

    tokens = tokenizer.encode(text)
    chunks = []
    start_index = 0
    chunk_index = 0
    total_tokens = len(tokens)

    while start_index < total_tokens:
        end_index = min(start_index + chunk_size, total_tokens)
        chunk_tokens = tokens[start_index:end_index]
        chunk_text = tokenizer.decode(chunk_tokens)

        # Basic metadata - page number requires more advanced PDF parsing
        chunks.append({
            "index": chunk_index,
            "content": chunk_text,
            "token_count": len(chunk_tokens),
            "page_number": -1 # Placeholder - requires library like pdfminer.six for better mapping
        })

        start_index += chunk_size - chunk_overlap
        if start_index >= end_index: # Prevent infinite loops if overlap >= size
             start_index = end_index # Move to the next non-overlapping position

        chunk_index += 1

    return chunks

def process_and_embed_document(uploaded_file, rag_config):
    """Processes a single uploaded PDF: extracts text, chunks, embeds, and stores in FAISS."""
    print(f"  [Process Doc] Top of function for '{uploaded_file.name}'") # DEBUG

    # Ensure vector store is initialized in session state 
    if st.session_state.faiss_index is None:
        print(f"  [Process Doc] Session state faiss_index is None. Initializing...") # DEBUG
        load_vector_store()
        print(f"  [Process Doc] Vector store initialized in session state.") # DEBUG
    else:
        # If already initialized, just log its current state
        print(f"  [Process Doc] Using existing session state faiss_index (Size: {st.session_state.faiss_index.ntotal}).")

    # Check initialization status
    if st.session_state.faiss_index is None:
        print(f"  [Process Doc] Exiting: faiss_index is None after initialization.") # DEBUG
        st.error("Vector store could not be initialized. Cannot process document.")
        return None

    print(f"  [Process Doc] Index check passed. Index has {st.session_state.faiss_index.ntotal} vectors.") # DEBUG
    start_time = time.time()
    st.info(f"Processing {uploaded_file.name}...")

    # 1. Generate unique ID for the document
    doc_id = generate_doc_id(uploaded_file.name)
    print(f"  [Process Doc] Generated doc_id: {doc_id}") # DEBUG

    # 2. Extract text
    print(f"  [Process Doc] Extracting text...") # DEBUG
    text = extract_text_from_pdf(uploaded_file)
    if not text:
        print(f"  [Process Doc] Exiting: Text extraction failed.") # DEBUG
        st.warning(f"Could not extract text from {uploaded_file.name}. Skipping.")
        return None
    print(f"  [Process Doc] Text extracted successfully ({len(text)} chars).") # DEBUG

    # 3. Chunk text
    print(f"  [Process Doc] Chunking text...") # DEBUG
    tokenizer = get_tokenizer()
    chunks = chunk_text(text, rag_config['chunk_size'], rag_config['chunk_overlap'], tokenizer)
    if not chunks:
        print(f"  [Process Doc] Exiting: No chunks generated.") # DEBUG
        st.warning(f"No text chunks generated for {uploaded_file.name}. Skipping.")
        return None
    print(f"  [Process Doc] Text chunked successfully ({len(chunks)} chunks).") # DEBUG

    # 4. Prepare data for embedding
    chunk_contents = [c['content'] for c in chunks]
    chunk_ids = [generate_chunk_id(doc_id, c['index']) for c in chunks]

    print(f"  [Process Doc] Generating embeddings...") # DEBUG
    st.info(f"Generating embeddings for {len(chunks)} chunks in {uploaded_file.name}...")
    embedding_model = get_embedding_model()
    try:
        embeddings = embedding_model.encode(chunk_contents, show_progress_bar=False)
        faiss.normalize_L2(embeddings)
        embeddings = embeddings.astype(np.float32)
        print(f"  [Process Doc] Embeddings generated successfully, shape: {embeddings.shape}") # DEBUG
    except Exception as e:
        print(f"  [Process Doc] Exiting: Embedding generation failed: {e}") # DEBUG
        st.error(f"Failed to generate embeddings for {uploaded_file.name}: {e}")
        return None

    # 5. Store in Vector Database (FAISS) and Metadata Map
    print(f"  [Process Doc] Storing chunks in FAISS...") # DEBUG
    try:
        num_new_chunks = len(chunks)
        faiss_ids_for_doc = np.arange(st.session_state.next_faiss_id, 
                                       st.session_state.next_faiss_id + num_new_chunks).astype('int64')

        # --- Corrected and more specific check ---
        if embeddings.shape[0] == 0:
             print(f"  [Process Doc] Exiting: Embeddings array is empty.") # DEBUG
             st.error(f"Embeddings array is empty for {uploaded_file.name}. Skipping add.")
             return None
        if embeddings.shape[1] != rag_config['embedding_dim']:
             print(f"  [Process Doc] Exiting: Embeddings dimension mismatch. Expected {rag_config['embedding_dim']}, got {embeddings.shape[1]}.") # DEBUG
             st.error(f"Embeddings dimension mismatch for {uploaded_file.name}. Expected {rag_config['embedding_dim']}, got {embeddings.shape[1]}. Skipping add.")
             return None
        # -------------------------------------------

        print(f"  [Process Doc] Calling faiss_index.add_with_ids...") # DEBUG
        st.session_state.faiss_index.add_with_ids(embeddings, faiss_ids_for_doc)
        print(f"  [Process Doc] Added {num_new_chunks} vectors. New total: {st.session_state.faiss_index.ntotal}")

        total_doc_tokens = 0
        for i, chunk_info in enumerate(chunks):
            faiss_id = int(faiss_ids_for_doc[i])
            chunk_id = chunk_ids[i]
            chunk_token_count = chunk_info['token_count']
            total_doc_tokens += chunk_token_count

            chunk_metadata = {
                "chunkId": chunk_id,
                "document_id": doc_id,
                "document_name": uploaded_file.name,
                "chunk_index": chunk_info['index'],
                "page_number": chunk_info['page_number'],
                "token_count": chunk_token_count,
                "content": chunk_info['content']
            }
            st.session_state.metadata_map[faiss_id] = chunk_metadata
            st.session_state.chunk_id_to_faiss_id[chunk_id] = faiss_id

        st.session_state.next_faiss_id += num_new_chunks
        print(f"  [Process Doc] Metadata maps updated. Next FAISS ID: {st.session_state.next_faiss_id}") # DEBUG

        print(f"  [Process Doc] Calling save_vector_store...") # DEBUG
        save_vector_store()
        print(f"  [Process Doc] save_vector_store returned.") # DEBUG
        st.session_state.vector_store_initialized = True

    except Exception as e:
        print(f"  [Process Doc] Exiting: Failed during storage phase: {e}") # DEBUG
        st.error(f"Failed to store chunks for {uploaded_file.name} in vector store: {e}")
        return None

    # 6. Update session state with document metadata
    print(f"  [Process Doc] Updating session state with doc metadata...") # DEBUG
    doc_metadata = {
        "id": doc_id,
        "name": uploaded_file.name,
        "size": uploaded_file.size,
        "uploadDate": get_current_timestamp(),
        "tokenCount": total_doc_tokens,
        "chunk_ids": chunk_ids
    }
    st.session_state.documents[doc_id] = doc_metadata
    print(f"  [Process Doc] Session state updated for doc_id: {doc_id}") # DEBUG

    end_time = time.time()
    st.success(f"Successfully processed {uploaded_file.name} ({len(chunks)} chunks, {total_doc_tokens} tokens) in {end_time - start_time:.2f} seconds.")
    print(f"  [Process Doc] === Successfully finished processing {uploaded_file.name} ===") # DEBUG
    return doc_id # Successful completion returns the doc_id


def delete_document_from_store(doc_id):
    """Removes a document and its chunks from FAISS and session state."""
    if st.session_state.faiss_index is None or not isinstance(st.session_state.faiss_index, faiss.IndexIDMap2):
         st.error("Cannot delete: FAISS index not initialized or does not support ID-based operations.")
         # Load store might fix this if it's just not loaded yet
         load_vector_store()
         if st.session_state.faiss_index is None or not isinstance(st.session_state.faiss_index, faiss.IndexIDMap2):
             st.error("Reloading store did not help. Deletion failed.")
             return

    if doc_id not in st.session_state.documents:
        st.warning(f"Document ID {doc_id} not found for deletion.")
        return

    doc_info = st.session_state.documents[doc_id]
    chunk_ids_to_delete = doc_info.get("chunk_ids", []) # Our generated chunk IDs

    if not chunk_ids_to_delete:
        st.warning(f"No chunk IDs found for document {doc_info['name']}. Cannot delete from vector store.")
    else:
        faiss_ids_to_remove = []
        missing_faiss_id_count = 0
        for chunk_id in chunk_ids_to_delete:
            faiss_id = st.session_state.chunk_id_to_faiss_id.get(chunk_id)
            if faiss_id is not None:
                faiss_ids_to_remove.append(faiss_id)
            else:
                 missing_faiss_id_count += 1
        
        if missing_faiss_id_count > 0:
            st.warning(f"Could not find FAISS ID mapping for {missing_faiss_id_count} chunks of {doc_info['name']}.")

        if not faiss_ids_to_remove:
             st.warning(f"No corresponding FAISS IDs found for chunks of {doc_info['name']}. Cannot delete from index.")
        else:
            try:
                print(f"Attempting to remove {len(faiss_ids_to_remove)} FAISS IDs: {faiss_ids_to_remove}")
                remove_count = st.session_state.faiss_index.remove_ids(np.array(faiss_ids_to_remove, dtype=np.int64))
                print(f"Successfully removed {remove_count} vectors from FAISS index.")
                if remove_count != len(faiss_ids_to_remove):
                     st.warning(f"FAISS reported removing {remove_count} IDs, but expected {len(faiss_ids_to_remove)}.")

                # Clean up metadata maps
                for chunk_id in chunk_ids_to_delete:
                    faiss_id = st.session_state.chunk_id_to_faiss_id.pop(chunk_id, None)
                    if faiss_id is not None:
                        st.session_state.metadata_map.pop(faiss_id, None)
                
                st.info(f"Removed {remove_count} chunks for {doc_info['name']} from vector store.")
                # Persist changes
                save_vector_store()

            except Exception as e:
                st.error(f"Error deleting chunks for {doc_info['name']} from vector store: {e}")
                # Decide if we should still remove from session state - probably yes
                # For now, we proceed to remove from session state regardless

    # Remove from session state
    del st.session_state.documents[doc_id]
    st.success(f"Removed document {doc_info['name']} from the application.")
    # Reset selected docs if the deleted doc was selected
    if doc_id in st.session_state.selected_docs_for_query:
        st.session_state.selected_docs_for_query.remove(doc_id)

    # Check if any documents remain to determine vector_store_initialized status
    # Use the index directly as the source of truth
    st.session_state.vector_store_initialized = (st.session_state.faiss_index is not None 
                                                and st.session_state.faiss_index.ntotal > 0) 
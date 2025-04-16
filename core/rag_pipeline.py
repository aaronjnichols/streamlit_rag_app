import streamlit as st
import anthropic
import os
import numpy as np
import faiss
from .utils import LLM_MODEL_NAME, EMBEDDING_DIM
from .document_processor import get_embedding_model

# --- Debugging Flag ---
DEBUG_RETRIEVAL = True 

@st.cache_resource
def get_anthropic_client(api_key: str):
    """Initializes and caches the Anthropic client for a given API key."""
    if not api_key:
        st.error("Anthropic API Key is missing.")
        return None
    try:
        if DEBUG_RETRIEVAL: print(f"Initializing Anthropic client (key ending with ...{api_key[-4:]})")
        client = anthropic.Anthropic(api_key=api_key)
        # Optional: Test the key with a simple call, but be mindful of costs/quotas
        # client.count_tokens("test")
        if DEBUG_RETRIEVAL: print("Anthropic client initialized successfully.")
        return client
    except Exception as e:
        st.error(f"Failed to initialize Anthropic client: {e}")
        if DEBUG_RETRIEVAL: print(f"Anthropic client initialization failed: {e}")
        return None

def retrieve_relevant_chunks(query, selected_doc_ids, rag_config):
    """Retrieves relevant chunks from FAISS based on the query and selected documents."""
    if DEBUG_RETRIEVAL: print(f"\n--- Starting Retrieval ---")
    if DEBUG_RETRIEVAL: print(f"Query: '{query}'")
    if DEBUG_RETRIEVAL: print(f"Selected Doc IDs: {selected_doc_ids}")
    if DEBUG_RETRIEVAL: print(f"RAG Config: {rag_config}")

    if "faiss_index" not in st.session_state or st.session_state.faiss_index is None:
         st.error("Vector store has not been initialized. Cannot retrieve chunks.")
         if DEBUG_RETRIEVAL: print("Session state faiss_index is None.")
         return []
    if DEBUG_RETRIEVAL: print(f"Using session state FAISS index (Size: {st.session_state.faiss_index.ntotal}) and metadata map (Size: {len(st.session_state.metadata_map)}).")

    if not st.session_state.vector_store_initialized or st.session_state.faiss_index.ntotal == 0:
        st.warning("Vector store not initialized or empty. Please upload documents.")
        if DEBUG_RETRIEVAL: print(f"Store not ready. Initialized: {st.session_state.vector_store_initialized}, Index total: {st.session_state.faiss_index.ntotal}")
        return []
    if not selected_doc_ids:
        st.warning("No documents selected for query.")
        if DEBUG_RETRIEVAL: print("No documents selected.")
        return []

    if not st.session_state.metadata_map:
         st.warning("Metadata map is empty. Cannot retrieve chunk details. Re-processing documents might be needed.")
         return []

    embedding_model = get_embedding_model()

    try:
        query_embedding = embedding_model.encode([query])[0]
        query_embedding_normalized = np.float32(query_embedding)
        faiss.normalize_L2(query_embedding_normalized.reshape(1, -1))
        query_embedding_normalized = query_embedding_normalized.reshape(1, -1)
        if DEBUG_RETRIEVAL: print(f"Generated query embedding shape: {query_embedding_normalized.shape}")
    except Exception as e:
        st.error(f"Failed to generate query embedding: {e}")
        if DEBUG_RETRIEVAL: print(f"Embedding error: {e}")
        return []

    k = rag_config['max_chunks']
    similarity_threshold = rag_config['similarity_threshold']

    search_k = k * 5 if selected_doc_ids != ["all"] else k
    if DEBUG_RETRIEVAL: print(f"Performing FAISS search: k={k}, search_k={search_k}, threshold={similarity_threshold}")
    
    try:
        scores, faiss_ids = st.session_state.faiss_index.search(query_embedding_normalized, search_k)
        
        scores = scores[0]
        faiss_ids = faiss_ids[0] 
        if DEBUG_RETRIEVAL: print(f"FAISS Raw Results - Scores: {scores}")
        if DEBUG_RETRIEVAL: print(f"FAISS Raw Results - FAISS IDs: {faiss_ids}")

    except Exception as e:
        st.error(f"Error querying FAISS index: {e}")
        if DEBUG_RETRIEVAL: print(f"FAISS search error: {e}")
        return []

    retrieved_chunks = []
    valid_results_count = 0
    if DEBUG_RETRIEVAL: print("--- Processing FAISS Results ---")
    for i in range(len(faiss_ids)):
        faiss_id = int(faiss_ids[i])
        score = float(scores[i])

        if DEBUG_RETRIEVAL: print(f"\nProcessing Result {i+1}: FAISS ID={faiss_id}, Score={score:.4f}")

        if faiss_id == -1:
            if DEBUG_RETRIEVAL: print(" Skipping FAISS ID -1")
            continue 
            
        valid_results_count += 1

        chunk_metadata = st.session_state.metadata_map.get(faiss_id)

        if not chunk_metadata:
            st.warning(f"Metadata not found for FAISS ID {faiss_id}. Skipping result.")
            if DEBUG_RETRIEVAL: print(f"  Metadata Check: FAILED - Metadata not found for FAISS ID {faiss_id}")
            continue
        else:
             if DEBUG_RETRIEVAL: print(f"  Metadata Check: SUCCESS - Found metadata for FAISS ID {faiss_id} (Doc: {chunk_metadata.get('document_name')}, ChunkID: {chunk_metadata.get('chunkId')})")

        if score < similarity_threshold:
            if DEBUG_RETRIEVAL: print(f"  Threshold Check: FAILED - Score {score:.4f} < {similarity_threshold}")
            continue
        else:
            if DEBUG_RETRIEVAL: print(f"  Threshold Check: PASSED - Score {score:.4f} >= {similarity_threshold}")

        doc_id = chunk_metadata.get("document_id")
        if selected_doc_ids != ["all"] and doc_id not in selected_doc_ids:
            if DEBUG_RETRIEVAL: print(f"  Document Filter: FAILED - Doc ID '{doc_id}' not in selected list {selected_doc_ids}")
            continue
        else:
             if DEBUG_RETRIEVAL: print(f"  Document Filter: PASSED - Doc ID '{doc_id}' matches selection {selected_doc_ids}")

        chunk_data = {
            "documentId": doc_id,
            "documentName": chunk_metadata.get("document_name", "N/A"),
            "chunkId": chunk_metadata.get("chunkId", "N/A"),
            "pageNumber": chunk_metadata.get("page_number", -1),
            "content": chunk_metadata.get("content", ""),
            "confidenceScore": score
        }
        retrieved_chunks.append(chunk_data)
        if DEBUG_RETRIEVAL: print(f"  >>> Added Chunk to final list: {chunk_data['chunkId']}")

        if len(retrieved_chunks) >= k:
            if DEBUG_RETRIEVAL: print(f"Reached target number of chunks ({k}). Stopping processing.")
            break
            
    if valid_results_count == 0:
        st.info("FAISS search returned no valid results.")
        if DEBUG_RETRIEVAL: print("FAISS search returned no valid results (IDs were -1 or loop didn't run).")

    st.success(f"Retrieved {len(retrieved_chunks)} relevant chunks after filtering.")
    if DEBUG_RETRIEVAL: print(f"--- Returning {len(retrieved_chunks)} Chunks ---")
    return retrieved_chunks

def generate_llm_response(query, retrieved_chunks, rag_config):
    """Generates a response from the LLM based on the query and retrieved chunks."""
    # --- Get API Key from Session State ---
    api_key = st.session_state.get("api_key")
    if not api_key:
         st.error("Anthropic API Key not found in session state. Please enter it in the sidebar.")
         return "Error: API Key missing.", []

    # --- Get Client using the key from session state ---
    llm_client = get_anthropic_client(api_key) # Pass the key here
    if not llm_client:
        # Error is already displayed by get_anthropic_client
        return "Error: LLM client could not be initialized.", []

    if not retrieved_chunks:
        if DEBUG_RETRIEVAL: print("generate_llm_response received no retrieved chunks.")
        return "I couldn't find relevant information in the selected documents to answer your query.", []

    context = """Relevant information from documents:
"""
    citations_used = []
    if DEBUG_RETRIEVAL: print("--- Constructing LLM Context ---")
    for i, chunk in enumerate(retrieved_chunks):
        context += f"--- Source {i+1} ---\
"
        context += f"Document: {chunk['documentName']}\
"
        if chunk['pageNumber'] != -1:
             context += f"Page: {chunk['pageNumber']}\
"
        context += f"Content: {chunk['content']}\
"
        context += "---\n\n"
        citations_used.append(chunk)
        if DEBUG_RETRIEVAL: print(f" Added chunk {chunk['chunkId']} (Score: {chunk['confidenceScore']:.4f}) to context.")

    prompt = f"""{context}
Based *only* on the provided sources above, answer the following question.
Cite the sources you used by number (e.g., [Source 1], [Source 2], etc.) after the relevant sentence or paragraph.
If the answer cannot be found in the sources, state that clearly.

Question: {query}

Answer:"""

    if DEBUG_RETRIEVAL: print(f"Sending request to LLM ({rag_config['llm_model']})...")

    try:
        # --- Added Debugging --- 
        masked_key = f"{api_key[:5]}...{api_key[-4:]}" if len(api_key) > 9 else api_key
        model_to_use = rag_config['llm_model']
        print(f"[DEBUG] Making Anthropic call with: Model='{model_to_use}', Key='{masked_key}'")
        # --- End Debugging ---

        message = llm_client.messages.create(
            model=model_to_use, # Use the variable defined above
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        ai_response_content = message.content[0].text
        st.success("LLM response received.")
        if DEBUG_RETRIEVAL: print("LLM response received successfully.")
        return ai_response_content, citations_used

    except anthropic.AuthenticationError:
         st.error("Anthropic API Key is invalid or expired. Please check the key in the sidebar.")
         if DEBUG_RETRIEVAL: print("LLM API Error: AuthenticationError")
         # Clear the cached client for this invalid key
         get_anthropic_client.clear() # Requires the function itself, not the instance
         return "Error: Invalid API Key.", []
    except Exception as e:
        st.error(f"Error contacting LLM: {e}")
        if DEBUG_RETRIEVAL: print(f"LLM API error: {e}")
        return f"Error generating response: {e}", [] 
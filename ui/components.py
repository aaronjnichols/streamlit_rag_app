import streamlit as st
from core.document_processor import delete_document_from_store

def display_document_list(documents):
    """Displays the list of uploaded documents with delete buttons."""
    st.subheader("Available Documents")
    if not documents:
        st.caption("No documents uploaded yet.")
        return

    for doc_id, doc_meta in list(documents.items()): # Use list() for safe iteration while deleting
        col1, col2, col3 = st.columns([4, 2, 1])
        with col1:
            st.text(f"{doc_meta.get('name', 'Unknown Name')}")
        with col2:
            token_count = doc_meta.get('tokenCount', 0)
            st.caption(f"~{token_count:,} tokens") # Formatted token count
        with col3:
            # Unique key using doc_id prevents issues with multiple delete buttons
            if st.button("❌", key=f"delete_{doc_id}", help=f"Delete {doc_meta.get('name', '')}"):
                with st.spinner(f"Deleting {doc_meta.get('name', '')}..."):
                    delete_document_from_store(doc_id)
                # After deletion, stop the current run and rerun to reflect the change immediately
                # This prevents potential errors if the deleted doc_id is used later in the same script run
                st.rerun()

def display_citations(citations):
    """Displays the source citations for an AI response."""
    if not citations:
        return

    with st.expander("View Sources"): # Changed label for clarity
        for i, citation in enumerate(citations):
            st.markdown(f"**Source {i+1}: {citation['documentName']}**")
            # Display page number only if valid (> -1)
            if citation['pageNumber'] != -1:
                st.caption(f"Page: {citation['pageNumber']} | Confidence: {citation['confidenceScore']:.2f}")
            else:
                st.caption(f"Confidence: {citation['confidenceScore']:.2f}")

            # Use st.text_area for scrollable content if chunks are long
            st.text_area(
                label=f"Chunk Content {i+1}",
                value=citation['content'], 
                height=100, 
                key=f"citation_{citation['chunkId']}_{i}",
                disabled=True # Make it read-only
            )
            st.markdown("---*", unsafe_allow_html=True) # Divider 
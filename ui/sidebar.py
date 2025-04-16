import streamlit as st
from core.document_processor import process_and_embed_document, get_embedding_model, load_vector_store
from core.utils import get_config_for_conversation
from .components import display_document_list
import time # Added for debug timestamp
import os # Added for path operations if needed

def display_sidebar():
    """Renders the sidebar UI elements and handles file uploads."""
    with st.sidebar:
        st.title("📄 PDF Chat RAG")
        st.markdown("---*", unsafe_allow_html=True)

        # --- Conversation Management (Placeholder) ---
        st.header("Conversations")
        # TODO: Implement conversation selection, creation, deletion
        # Example: convo_list = list(st.session_state.conversations.keys())
        # selected_convo = st.selectbox("Select Conversation", convo_list, index=0 if convo_list else -1)
        # if st.button("New Conversation"): create_new_conversation()
        # st.session_state.current_conversation_id = selected_convo # Update current convo
        st.caption("Conversation management coming soon.")
        st.markdown("---*", unsafe_allow_html=True)

        # --- RAG Configuration (Placeholder) ---
        st.header("RAG Settings")
        # TODO: Implement selection and editing of RAG configurations
        # Example: config_names = list(st.session_state.rag_configs.keys())
        # selected_config_name = st.selectbox("Select Config", config_names)
        # current_config = st.session_state.rag_configs[selected_config_name]
        # Display/edit config params like chunk size, overlap etc.
        st.caption("Configuration options coming soon.")
        st.markdown("---*", unsafe_allow_html=True)

        # --- Document Management ---
        st.header("Document Management")
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            accept_multiple_files=True,
            type="pdf",
            key="file_uploader",
            help="Upload one or more PDF files to process and chat with."
        )

        # Check if there are files in the uploader widget
        if uploaded_files:
            # Use a separate flag to track if processing *should* happen for *this specific batch*
            # We reset this only when the uploader itself changes OR after processing completes.
            # This uses the uploader's state implicitly. A more robust way might involve session state keys.
            if 'processed_file_set' not in st.session_state:
                 st.session_state.processed_file_set = set()

            # Identify files in the current upload batch that haven't been processed based on name
            files_to_process = []
            skipped_files = []
            existing_doc_names = {meta['name'] for meta in st.session_state.documents.values()}
            print(f"\n--- Sidebar: Checking Uploaded Files ({time.strftime('%H:%M:%S')}) ---")
            print(f"Existing doc names in session state: {existing_doc_names}")

            for uploaded_file in uploaded_files:
                 print(f" Checking file: '{uploaded_file.name}', Size: {uploaded_file.size}") # DEBUG
                 if uploaded_file.name not in existing_doc_names:
                     # Add to list if name not found in already processed documents
                     files_to_process.append(uploaded_file)
                     print(f"  -> Marked '{uploaded_file.name}' for processing.") # DEBUG
                 else:
                     # Skip if name already exists in session state
                     skipped_files.append(uploaded_file.name)
                     print(f"  -> Skipping '{uploaded_file.name}' - Filename already exists in session state.") # DEBUG


            if skipped_files:
                 st.warning(f"Skipped already processed files: {', '.join(skipped_files)}")

            # Process only the files identified as new
            if files_to_process:
                num_files = len(files_to_process)
                st.session_state.processing_files = True # Flag to indicate processing is ongoing
                print(f"Attempting to process {num_files} new file(s)...")
                progress_bar = st.progress(0, text=f"Starting processing for {num_files} file(s)...")
                processed_count = 0

                with st.spinner("Processing documents..."):
                    active_rag_config = get_config_for_conversation(st.session_state.current_conversation_id)

                    for i, file_to_process in enumerate(files_to_process):
                        print(f" ==> Processing '{file_to_process.name}'...") # DEBUG
                        progress_text = f"Processing file {i+1}/{num_files}: {file_to_process.name}..."
                        progress_bar.progress((i + 1) / num_files, text=progress_text)

                        process_result_doc_id = process_and_embed_document(file_to_process, active_rag_config)

                        if process_result_doc_id:
                            processed_count += 1
                            print(f" ==> Successfully processed '{file_to_process.name}', new doc_id: {process_result_doc_id}") # DEBUG
                            # Add successfully processed filename to our set to avoid reprocessing *this batch* on rerun
                            st.session_state.processed_file_set.add(file_to_process.name)
                        else:
                            print(f" ==> Failed to process '{file_to_process.name}' (returned None)") # DEBUG


                progress_bar.empty()
                if processed_count > 0:
                    st.success(f"Finished processing {processed_count} new document(s).")
                    print(f"--- Sidebar: Finished Processing. Processed {processed_count} new file(s). Rerunning. ---") # DEBUG
                    # Clear the processed set for the *next* upload batch
                    st.session_state.processed_file_set = set()
                    st.session_state.processing_files = False # Reset flag
                    st.rerun() # Rerun to update UI and clear uploader potentially
                else:
                     st.info("No new documents were successfully processed.")
                     print(f"--- Sidebar: Finished Processing. No new files processed. ---") # DEBUG
                     st.session_state.processing_files = False # Reset flag even if nothing new processed
            # If no files to process were identified, clear the flag
            elif not files_to_process and not st.session_state.processing_files:
                 # This case might not be strictly necessary if the flag logic works, but safe fallback
                 pass


        # Display the list of currently available documents (regardless of upload state)
        display_document_list(st.session_state.documents) 
import streamlit as st
from .components import display_citations
import time

# --- Conversation Management --- #

def create_new_conversation():
    """Creates a new conversation entry in session state."""
    convo_id = f"convo_{int(time.time())}"
    # Find a unique name
    i = 1
    existing_names = {c.get('name', '') for c in st.session_state.conversations.values()}
    while f"Conversation {i}" in existing_names:
        i += 1
    convo_name = f"Conversation {i}"

    st.session_state.conversations[convo_id] = {
        "id": convo_id,
        "name": convo_name,
        "messages": [], # Store as list of dicts: {"role": "user/assistant", "content": "...", "citations": []}
        "rag_config_id": "default" # Link to RAG config
    }
    st.session_state.current_conversation_id = convo_id
    st.success(f"Created new conversation: {convo_name}")
    # No rerun here, handled in app.py or sidebar potentially

def add_message_to_current_conversation(role, content, citations=None, selected_docs=None):
    """Adds a message to the currently active conversation."""
    convo_id = st.session_state.current_conversation_id
    if convo_id and convo_id in st.session_state.conversations:
        message = {"role": role, "content": content, "timestamp": time.time()}
        if citations:
            # Ensure citations are serializable if needed, they should be dicts here
            message["citations"] = citations
        if selected_docs:
            message["selected_docs"] = selected_docs # Store which docs were used for the user query
        st.session_state.conversations[convo_id]["messages"].append(message)
    else:
        # This case should ideally be prevented by the UI flow
        print(f"Error: Attempted to add message to invalid convo_id '{convo_id}'")

# --- Message Display --- #

def display_message(message):
    """Displays a single chat message with citations if available."""
    role = message.get('role', 'assistant') # Default to assistant if role missing
    content = message.get('content', '')
    citations = message.get('citations', [])
    selected_docs = message.get('selected_docs', [])

    avatar_map = {"user": "👤", "assistant": "🤖"}
    with st.chat_message(role, avatar=avatar_map.get(role)):
        st.markdown(content)
        if role == "assistant" and citations:
            display_citations(citations)
        if role == "user" and selected_docs:
            display_used_documents_for_message(selected_docs)

def display_used_documents_for_message(selected_docs):
    """Displays the document context used for a specific user message."""
    doc_names = []
    if selected_docs == ["all"]:
        # Get all available document names
        doc_names = ["All Uploaded Documents"] # Simplified representation
        # More detailed: doc_names = [meta['name'] for meta in st.session_state.documents.values()]
    elif isinstance(selected_docs, list) and selected_docs:
        doc_names = [
            st.session_state.documents.get(doc_id, {}).get('name', f'ID: {doc_id}')
            for doc_id in selected_docs
            if doc_id in st.session_state.documents # Check if doc still exists
        ]

    if doc_names:
        # Display below the user message bubble
        st.caption(f"*Context: {", ".join(doc_names)}*", help="Documents used as context for this query.")

# --- Chat Interface Rendering --- #

def display_chat_interface():
    """Displays the main chat area, message history, and input."""

    if not st.session_state.current_conversation_id:
        st.info("Create a new conversation or select an existing one from the sidebar to start chatting.")
        if st.button("✨ Start New Conversation"):
             create_new_conversation()
             st.rerun()
        return None # Return None if no conversation is active

    convo_id = st.session_state.current_conversation_id
    conversation = st.session_state.conversations.get(convo_id)

    if not conversation:
        st.error(f"Error: Current conversation ID '{convo_id}' not found. Please select or create a conversation.")
        st.session_state.current_conversation_id = None # Reset to avoid repeated errors
        st.rerun()
        return None

    st.header(f"Chat: {conversation.get('name', 'Unnamed Conversation')}")

    # Display existing messages
    for message in conversation.get("messages", []):
        display_message(message)

    # --- Document Selection for Next Query --- #
    st.markdown("---*", unsafe_allow_html=True)
    st.subheader("Select Document Context for Next Query")

    available_docs = st.session_state.documents
    doc_options = {doc_id: meta['name'] for doc_id, meta in available_docs.items()}

    # Key definitions for widgets
    use_all_docs_key = f"use_all_docs_{convo_id}"
    multiselect_key = f"doc_selector_{convo_id}"

    # Determine the current state directly from session state (initialize if missing)
    if "selected_docs_for_query" not in st.session_state:
        st.session_state.selected_docs_for_query = ["all"] if available_docs else []
        
    current_selection = st.session_state.selected_docs_for_query
    is_using_all_docs = current_selection == ["all"]

    # Callback for the checkbox
    def checkbox_changed():
        if st.session_state[use_all_docs_key]: # Check the widget's current state in the callback
            st.session_state.selected_docs_for_query = ["all"]
        else:
            # When unchecking 'All', clear the selection. User must use multiselect.
            st.session_state.selected_docs_for_query = []

    use_all_docs_checkbox = st.checkbox(
        "Use All Uploaded Documents",
        value=is_using_all_docs, # Default value based on session state
        key=use_all_docs_key,
        help="Check this to use all available documents as context.",
        on_change=checkbox_changed # Update state when changed
    )

    # Determine default for multiselect based on session state (only relevant if 'use_all' is false)
    default_multiselect_selection = []
    if not is_using_all_docs and isinstance(current_selection, list):
        # Filter selection to only include currently available docs
        default_multiselect_selection = [doc_id for doc_id in current_selection if doc_id in doc_options]

    # Multiselect - only enabled if 'use_all' is not checked and docs are available
    show_multiselect = not use_all_docs_checkbox and available_docs

    # Callback for the multiselect
    def multiselect_changed():
        # Read the current value from the widget state inside the callback
        st.session_state.selected_docs_for_query = st.session_state[multiselect_key]

    if show_multiselect:
        # Display the multiselect widget
        st.multiselect(
            "Or, Select Specific Documents:",
            options=list(doc_options.keys()),
            format_func=lambda doc_id: doc_options.get(doc_id, f"Unknown ID: {doc_id}"),
            default=default_multiselect_selection, # Default based on filtered session state
            key=multiselect_key,
            help="Select one or more documents to use as context for your next query.",
            on_change=multiselect_changed # Update state when changed
        )
    elif not available_docs:
         st.caption("Upload documents using the sidebar to enable selection.")
         # Ensure state is empty if no docs are available
         if st.session_state.selected_docs_for_query != []:
             st.session_state.selected_docs_for_query = []
             st.rerun() # Rerun if we just cleared the state because docs disappeared

    st.markdown("---*", unsafe_allow_html=True)

    # --- Chat Input --- #
    prompt = st.chat_input("Ask a question about the selected documents...")

    # Important: The state (st.session_state.selected_docs_for_query) is now updated
    # by the callbacks *before* this function returns the prompt.
    # The RAG logic in app.py will read the correct, user-intended state.

    return prompt # Return the user's prompt if entered, otherwise None 
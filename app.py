import streamlit as st
import torch
import os

# --- Workaround for Streamlit/Torch __path__ issue ---
# See: https://github.com/pytorch/pytorch/issues/11201
# And: https://discuss.streamlit.io/t/error-in-torch-with-streamlit/90908
try:
    # Try setting the path explicitly if it exists
    if hasattr(torch, 'classes') and hasattr(torch.classes, '__file__'):
         torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
    # Fallback: Set to an empty list if the above fails or attributes don't exist
    elif hasattr(torch, 'classes'):
         torch.classes.__path__ = []
    print("Applied torch.classes.__path__ workaround.")
except Exception as e:
    print(f"Could not apply torch.classes.__path__ workaround: {e}")
# --- End Workaround ---

# Set page config first
st.set_page_config(layout="wide", page_title="PDF Chat RAG", page_icon="📄")

from core.utils import initialize_session_state, get_config_for_conversation
from core.rag_pipeline import retrieve_relevant_chunks, generate_llm_response, get_anthropic_client
from ui.sidebar import display_sidebar
from ui.chat_interface import display_chat_interface, add_message_to_current_conversation

# --- Configuration & Sidebar ---
st.title("📄 PDF Chat RAG Application")

st.sidebar.header("Configuration")

# API Key Input
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("ANTHROPIC_API_KEY", "") # Load from env first

api_key_input = st.sidebar.text_input(
    "Anthropic API Key",
    type="password",
    value=st.session_state.api_key,
    help="Get your API key from https://console.anthropic.com/settings/keys"
)

# Update session state if input changes
if api_key_input != st.session_state.api_key:
    st.session_state.api_key = api_key_input.strip()
    # Optionally, clear previous chat/data if key changes mid-session
    # Clear the cached Anthropic client if the key changes
    if 'get_anthropic_client' in globals():
        get_anthropic_client.clear()
    # st.session_state.messages = []
    # st.session_state.vector_store = None # Example

# Check if API key is provided before proceeding
# Also strip the key currently in session state just in case it came from os.getenv
if st.session_state.api_key:
    st.session_state.api_key = st.session_state.api_key.strip()

if not st.session_state.api_key:
    st.info("Please add your Anthropic API key in the sidebar to continue.")
    st.stop() # Stop execution if no key

# --- Initialize LLM Client (using the key from session state) ---
try:
    # Example: Assuming you have a function or class that takes the key
    # from anthropic import Anthropic # Make sure this is imported
    # client = Anthropic(api_key=st.session_state.api_key)
    # Replace the line above with your actual client initialization
    pass # Placeholder for your actual client init using st.session_state.api_key
except Exception as e:
    st.error(f"Failed to initialize Anthropic client: {e}")
    st.stop()

def main():
    """Main function to run the Streamlit application."""

    # --- Initialization --- 
    initialize_session_state() # Ensure all session state keys exist

    # --- UI Rendering --- 
    display_sidebar() # Render sidebar (handles uploads, doc list, etc.)

    # Render main chat area and get user input
    user_prompt = display_chat_interface() # Displays history, doc selection, input box

    # --- Handle User Input and RAG --- 
    if user_prompt:
        current_convo_id = st.session_state.current_conversation_id
        if not current_convo_id:
            st.error("Cannot process query: No active conversation.")
            return

        # 1. Get selected documents for this query from session state
        selected_docs = st.session_state.selected_docs_for_query
        if not selected_docs:
            st.warning("Please select documents to query before asking a question.")
            return # Stop processing if no documents are selected

        # 2. Add user message to conversation history
        add_message_to_current_conversation("user", user_prompt, selected_docs=selected_docs)

        # Display progress/spinner while processing
        with st.spinner("Thinking..."):
            # 3. Get the appropriate RAG config
            # TODO: Allow selecting config per conversation in the future
            active_rag_config = get_config_for_conversation(current_convo_id)

            # 4. Retrieve relevant chunks
            retrieved_chunks = retrieve_relevant_chunks(
                user_prompt,
                selected_docs,
                active_rag_config
            )

            # 5. Generate LLM response
            ai_response, citations = generate_llm_response(
                user_prompt,
                retrieved_chunks,
                active_rag_config
            )

        # 6. Add AI response to conversation history
        add_message_to_current_conversation("assistant", ai_response, citations=citations)

        # 7. Rerun Streamlit to update the chat display with the new messages
        st.rerun()

if __name__ == "__main__":
    main() 
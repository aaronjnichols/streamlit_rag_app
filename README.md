# PDF Chat RAG Application

A Streamlit application to chat with PDF documents using Retrieval Augmented Generation (RAG).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd rag_app
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Key:**
    - Set the `ANTHROPIC_API_KEY` environment variable with your Anthropic API key.

5.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## Project Structure

```
rag_app/
├── app.py               # Main Streamlit application entry point
├── core/                # Core application logic modules
│   ├── __init__.py
│   ├── document_processor.py
│   ├── rag_pipeline.py
│   └── utils.py
├── ui/                  # Streamlit UI component modules
│   ├── __init__.py
│   ├── sidebar.py
│   ├── chat_interface.py
│   └── components.py
├── data/                # Directory for persistent data (ChromaDB)
│   └── chroma_db/
├── requirements.txt     # Python package dependencies
└── README.md            # Project description
``` 
# Doc-Query

# RAG-Based PDF Query System

This project implements a Retrieval-Augmented Generation (RAG) based system
to query PDF documents using natural language.

## Features
- Multi PDF document processing
- Natural language question answering
- Retrieval-Augmented Generation (RAG)
- Simple Python-based implementation
- Automatic Question Generation from Uploaded PDFs
- Context Aware retriever
- Sessions stored and can be exported as JSON file
- Demo Mode (if API not available)

## Tech Stack
- Python
- LangChain
- HuggingFace Embeddings
- Streamlit (for UI)
- GROQ 
- SQLite

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Run the application:
   python app.py

## Note
API keys and environment variables are not included for security reasons.

This application requires the following API keys to function correctly:

- Hugging Face API Token  
  Used for generating embeddings and language model inference.

- LLM Provider API Key (e.g., OpenAI / Groq / compatible LLM service)  
  Used for generating natural language responses.

These API keys must be stored securely in a `.env` file and are intentionally
not included in this repository for security reasons.

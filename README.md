# Government Schemes RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions
about government schemes using information from official PDF documents.

## Problem Statement

A chatbot that answers farmer scheme questions from official government PDFs
and provides the source document and page number.

## Tech Stack

- Python
- LangChain
- ChromaDB
- Sentence Transformers
- PyPDF
- Streamlit
- Groq / Gemini

## Project Status

Phase 1 - Setup and Planning

## Project Structure

```text
rag-chatbot/
├── data/
│   └── documents/
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── chatbot.py
│   └── utils.py
├── tests/
│   └── __init__.py
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── main.py
└── README.md
```

## Important

- Put official PDF documents inside `data/documents/`.
- Keep API keys in `.env`.
- Never commit `.env` or `venv/` to GitHub.

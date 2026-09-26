\# 📚 Local RAG Chatbot



A knowledge-grounded AI chatbot built with Retrieval-Augmented Generation (RAG). The application allows users to upload PDF documents, stores their semantic representations in ChromaDB, retrieves relevant information using MMR, and generates answers using a local Llama 3.2 model through Ollama.



\## Features



\- 📄 Upload PDF documents through the web interface

\- 🔎 Semantic document retrieval

\- 🧠 MMR-based retrieval using ChromaDB

\- 🤖 Local Llama 3.2 3B LLM through Ollama

\- 🧩 Hugging Face sentence-transformer embeddings

\- 💬 Interactive Streamlit chat interface

\- 📚 Multiple documents in the knowledge base

\- 📌 Document and page-level source citations

\- 💾 Persistent ChromaDB vector store



\## Architecture



```text

&#x20;                User

&#x20;                 │

&#x20;                 ▼

&#x20;          Streamlit Chat UI

&#x20;                 │

&#x20;                 ▼

&#x20;            User Question

&#x20;                 │

&#x20;                 ▼

&#x20;            ChromaDB

&#x20;                 │

&#x20;           MMR Retrieval

&#x20;                 │

&#x20;                 ▼

&#x20;         Relevant PDF Chunks

&#x20;                 │

&#x20;                 ▼

&#x20;         Context + Question

&#x20;                 │

&#x20;                 ▼

&#x20;       Llama 3.2 3B / Ollama

&#x20;                 │

&#x20;                 ▼

&#x20;         Grounded Answer

&#x20;                 │

&#x20;                 ▼

&#x20;         Document Sources







Tech Stack

Python

LangChain

Streamlit

ChromaDB

Ollama

Llama 3.2 3B

Hugging Face Sentence Transformers

PyPDF

Project Structure

RAG\_ChatBot/

│

├── documents/

│   └── knowledge\_base.pdf

│

├── app.py

├── chat.py

├── ingest.py

├── retrieve.py

├── requirements.txt

├── README.md

└── .gitignore

How It Works

1\. Document Ingestion



PDF documents are loaded using PyPDFLoader and divided into smaller chunks using RecursiveCharacterTextSplitter.



2\. Embeddings



Each document chunk is converted into a numerical embedding using:



sentence-transformers/all-MiniLM-L6-v2

3\. Vector Database



The embeddings are stored in ChromaDB.



4\. Retrieval



When a user asks a question, ChromaDB retrieves relevant chunks using Maximal Marginal Relevance (MMR).



5\. Generation



The retrieved context is passed to the local:



Llama 3.2 3B



model through Ollama.



6\. Sources



The chatbot displays the document name and page number associated with retrieved chunks.


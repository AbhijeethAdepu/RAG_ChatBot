# 📚 Local RAG Chatbot

A knowledge-grounded AI chatbot built using **Retrieval-Augmented Generation (RAG)**.

The application allows users to upload PDF documents, convert their content into semantic embeddings, store them in **ChromaDB**, retrieve relevant information using **Maximal Marginal Relevance (MMR)**, and generate grounded responses using **Llama 3.2 3B** running locally through **Ollama**.

---

## ✨ Features

- 📄 Upload PDF documents through the web interface
- 🔎 Semantic document retrieval
- 🧠 MMR-based retrieval for relevant and diverse results
- 🤖 Local Llama 3.2 3B inference through Ollama
- 🧩 Hugging Face Sentence Transformer embeddings
- 💬 Interactive Streamlit chat interface
- 📚 Support for multiple documents
- 📌 Document and page-level source citations
- 💾 Persistent ChromaDB vector store

---

## 🏗️ Architecture

```text
                    User
                      │
                      ▼
              Streamlit Chat UI
                      │
                      ▼
                User Question
                      │
                      ▼
              ChromaDB Retrieval
                      │
                MMR Retrieval
                      │
                      ▼
             Relevant PDF Chunks
                      │
                      ▼
             Context + Question
                      │
                      ▼
             Llama 3.2 3B
                 via Ollama
                      │
                      ▼
              Grounded Answer
                      │
                      ▼
              Document Sources



🔄 How It Works


1. Document Ingestion

PDF documents are loaded using PyPDFLoader and divided into smaller chunks using RecursiveCharacterTextSplitter.

2. Embeddings

Each document chunk is converted into a numerical vector using:

sentence-transformers/all-MiniLM-L6-v2
3. Vector Database

The generated embeddings and document chunks are stored in ChromaDB for semantic retrieval.

4. Retrieval

When a user asks a question, the system searches the vector database and uses Maximal Marginal Relevance (MMR) to retrieve relevant and diverse document chunks.

5. Generation

The retrieved chunks are provided as context to:

Llama 3.2 3B

The model runs locally through Ollama and generates an answer based on the retrieved context.

6. Source Attribution

The application displays the source document and page number associated with the retrieved chunks.

🛠️ Tech Stack

Technology	Purpose
Python	Application development
LangChain	RAG pipeline and document processing
Streamlit	Web interface
ChromaDB	Vector database
Ollama	Local LLM runtime
Llama 3.2 3B	Language model
Hugging Face Sentence Transformers	Text embeddings
PyPDF	PDF document loading


📁 Project Structure

RAG_ChatBot/
│
├── documents/
│   └── knowledge_base.pdf
│
├── outputs/
│   ├── rag-chatbot-document-upload.png
│   ├── rag-chatbot-pdf-processing.png
│   ├── rag-chatbot-response.png
│   └── rag-chatbot-sources.png
│
├── app.py
├── chat.py
├── ingest.py
├── retrieve.py
├── requirements.txt
├── README.md
└── .gitignore
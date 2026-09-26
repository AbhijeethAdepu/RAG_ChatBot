from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DOCUMENTS_DIR = Path("documents")
VECTORSTORE_DIR = "vectorstore"


def load_documents():
    documents = []

    for pdf_file in DOCUMENTS_DIR.glob("*.pdf"):
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    if not documents:
        raise FileNotFoundError(
            "No PDF files found in the documents folder."
        )

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(documents)


def create_vector_store(chunks):
    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating Chroma vector database...")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR
    )


def main():
    print("Starting document ingestion...\n")

    documents = load_documents()
    print(f"Loaded {len(documents)} pages.")

    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    create_vector_store(chunks)

    print("\nVector database created successfully!")


if __name__ == "__main__":
    main()
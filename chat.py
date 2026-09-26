from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

VECTORSTORE_DIR = "vectorstore"


# Load embedding model once
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load vector database once
print("Loading vector database...")

vectorstore = Chroma(
    persist_directory=VECTORSTORE_DIR,
    embedding_function=embeddings,
)


# Load LLM once
print("Loading Llama 3.2...")

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


def ask_question(question):

    # Retrieve relevant chunks
    results = vectorstore.similarity_search(
        question,
        k=3
    )

    # Build context
    context = "\n\n".join(
        document.page_content
        for document in results
    )

    # Build prompt
    prompt = f"""
You are a helpful assistant answering questions about the provided document.

Use ONLY the information in the context below to answer the question.

If the answer cannot be found in the context, say:
"I couldn't find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate answer
    response = llm.invoke(prompt)

    return response.content, results


def main():

    print("\nRAG Chatbot")
    print("Ask questions about your PDF.")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question.strip():
            continue

        answer, results = ask_question(question)

        print("\n--- Answer ---\n")
        print(answer)

        print("\n--- Sources ---\n")

        for i, document in enumerate(results, start=1):

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get("page")

            if page is not None:
                page += 1

            print(f"Source {i}:")
            print(f"  File: {source}")
            print(f"  Page: {page}")

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
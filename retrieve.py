from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

VECTORSTORE_DIR = "vectorstore"


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
    )

    return vectorstore


def search_documents(question):
    vectorstore = load_vectorstore()

    results = vectorstore.max_marginal_relevance_search(
        question,
        k=3,
        fetch_k=10,
        lambda_mult=0.7
    )

    return results


def main():
    question = input("Ask a question about your PDF: ")

    results = search_documents(question)

    print("\n--- Retrieved Documents ---\n")

    for i, document in enumerate(results, start=1):
        print(f"Result {i}")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Page: {document.metadata.get('page')}")

        print(f"\n{document.page_content}")

        print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
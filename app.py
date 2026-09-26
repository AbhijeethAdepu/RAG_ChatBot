import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama



# Configuration


VECTORSTORE_DIR = "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama3.2:3b"



# Load resources


@st.cache_resource
def load_resources():

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Loading vector database...")

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
    )

    print("Loading Llama 3.2...")

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0
    )

    return vectorstore, llm


vectorstore, llm = load_resources()



# PDF Processing


def process_uploaded_pdf(uploaded_file):

    # Create temporary PDF file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:

        # Load PDF
        loader = PyPDFLoader(temp_path)

        documents = loader.load()

        # Split document into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        # Store original uploaded filename
        for chunk in chunks:
            chunk.metadata["source"] = uploaded_file.name

        # Add chunks to ChromaDB
        vectorstore.add_documents(chunks)

        return len(chunks)

    finally:

        # Delete temporary file
        os.remove(temp_path)



# RAG Question Answering


def ask_question(question):

    # Retrieve relevant chunks using MMR
    results = vectorstore.max_marginal_relevance_search(
        question,
        k=3,
        fetch_k=10,
        lambda_mult=0.7
    )

    # Build context from retrieved documents
    context = "\n\n".join(
        document.page_content
        for document in results
    )

    # Build prompt
    prompt = f"""
You are a helpful AI assistant answering questions using
the provided document context.

Use the context below as the primary source for your answer.

If the answer is clearly present in the context, explain it
accurately using the context.

If the answer cannot be found in the context, say:

"I couldn't find the answer in the uploaded documents."

Do not invent information and do not claim that information
came from a document when it did not.

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate answer
    response = llm.invoke(prompt)

    return response.content, results



# Streamlit Configuration


st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide"
)



# Header


st.title("📚 RAG Chatbot")

st.write(
    "Ask questions about your knowledge base and uploaded documents "
    "using Retrieval-Augmented Generation."
)



# Sidebar - PDF Upload


st.sidebar.header("📄 Add a Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file:

    st.sidebar.write(
        f"Selected: **{uploaded_file.name}**"
    )

    if st.sidebar.button("Add to Knowledge Base"):

        with st.sidebar.status(
            "Processing PDF...",
            expanded=True
        ):

            try:

                chunk_count = process_uploaded_pdf(
                    uploaded_file
                )

                st.sidebar.success(
                    f"Added {chunk_count} chunks!"
                )

            except Exception as e:

                st.sidebar.error(
                    f"Error processing PDF: {e}"
                )



# Chat History


if "messages" not in st.session_state:

    st.session_state.messages = []



# Display Previous Messages


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display sources for assistant messages
        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                with st.expander("📄 Sources"):

                    for source in sources:

                        st.write(
                            f"• {source['file']} — "
                            f"Page {source['page']}"
                        )


# Chat Input
 

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    
    # Display user message
     

    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    
    # Generate answer

    with st.chat_message("assistant"):

        with st.spinner("Searching documents and generating answer..."):

            try:

                answer, results = ask_question(question)

                st.markdown(answer)

                # Build source information
                sources = []

                for document in results:

                    source = document.metadata.get(
                        "source",
                        "Unknown"
                    )

                    page = document.metadata.get(
                        "page"
                    )

                    if page is not None:
                        page += 1

                    sources.append(
                        {
                            "file": source,
                            "page": page
                        }
                    )

                # Display sources

                if sources:

                    with st.expander("📄 Sources"):

                        for source in sources:

                            st.write(
                                f"• {source['file']} — "
                                f"Page {source['page']}"
                            )


                # Save assistant message

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )


            except Exception as e:

                error_message = (
                    f"Something went wrong: {e}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )
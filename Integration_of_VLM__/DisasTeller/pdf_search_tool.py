from crewai.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(script_dir, "Guidelines", "EMS98_Original_english__earthquake.pdf")
CHROMA_DB_DIR = os.path.join(script_dir, "gemini_chroma_db")

@tool
def offline_pdf_search_tool(query: str) -> str:
    """Semantic search over a local earthquake guidelines PDF using Gemini Embeddings."""

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

    # Use google gemini embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    # If DB exists, load it
    if os.path.exists(CHROMA_DB_DIR):
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embedding_model
        )
    else:
        # Else, load and index the PDF
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=CHROMA_DB_DIR
        )

    # Run search
    results = vectorstore.similarity_search(query, k=3)

    # Format output
    return "\n\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(results)])

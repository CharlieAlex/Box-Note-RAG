from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from app.config import settings


def get_vector_store():
    client = PersistentClient(path=settings.chroma_path)
    embeddings = OllamaEmbeddings(model=settings.embeddings_model)
    return Chroma(
        client=client,
        embedding_function=embeddings,
    )

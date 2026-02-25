from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from app.config import get_settings


def get_vector_store():
    client = PersistentClient(path=get_settings().chroma_path)
    embeddings = OllamaEmbeddings(model=get_settings().embeddings_model)
    return Chroma(
        client=client,
        embedding_function=embeddings,
    )

from functools import cache, lru_cache

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from .config import settings

EMBEDDINGS_MODEL = settings.embeddings_model
OLLAMA_MODEL = settings.ollama_model
CHROMA_PATH = settings.chroma_path


@cache
def get_retriever():
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OllamaEmbeddings(model=EMBEDDINGS_MODEL)
    )
    return vectorstore.as_retriever(search_kwargs={"k": 10})


@lru_cache
def get_llm():
    return ChatOllama(model=OLLAMA_MODEL, temperature=0)

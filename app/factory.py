from functools import cache, lru_cache

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
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


def get_bm25_retriever(k=10):
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OllamaEmbeddings(model=EMBEDDINGS_MODEL),
    )
    # 從 Chroma 取出所有文件來建立 BM25
    all_data = vectorstore._collection.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    return BM25Retriever.from_documents(docs, k=k)

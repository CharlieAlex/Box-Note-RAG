from pathlib import Path

import typer
from langchain_ollama import OllamaEmbeddings
from loguru import logger
from tqdm import tqdm

from app.config import get_settings
from app.retriever.loaders import load_documents
from app.retriever.splitters import splitter_mapping
from app.retriever.vector_store import get_vector_store

BATCH_SIZE = get_settings().batch_size
SPLITTER = get_settings().splitter
SPLITTER_ARGS = {
    "chunk_size": 400,
    "chunk_overlap": 50,
    "embeddings": OllamaEmbeddings(model=get_settings().embeddings_model),
    "breakpoint_threshold_type": "percentile"
}

app = typer.Typer()


@app.command()
def run_ingest(data_path: str = typer.Option(..., help="資料路徑")):
    # 1. 載入
    docs = load_documents(data_path)
    for doc in docs:
        # 把 source 路徑只留下文件名稱
        doc.metadata["source"] = Path(doc.metadata.get("source", "Unknown")).stem

    # 2. 切片
    splitter_func = splitter_mapping.get(SPLITTER)
    splitter = splitter_func(**SPLITTER_ARGS)
    chunks = splitter.split_documents(docs)

    # 3. 存入
    vectorstore = get_vector_store()

    with tqdm(total=len(chunks), desc="存入 Chroma") as pbar:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            vectorstore.add_documents(batch)
            pbar.update(len(batch))

    logger.success("✅ Ingestion 成功完成")


if __name__ == "__main__":
    app()

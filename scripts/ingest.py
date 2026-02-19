import typer
from loguru import logger
from tqdm import tqdm

from app.config import settings
from app.retriever.loaders import load_documents
from app.retriever.splitters import get_recursive_splitter
from app.retriever.vector_store import get_vector_store

BATCH_SIZE = settings.batch_size
app = typer.Typer()


@app.command()
def run_ingest(data_path: str = typer.Option(..., help="資料路徑")):
    # 1. 載入
    docs = load_documents(data_path)

    # 2. 切片
    splitter = get_recursive_splitter(chunk_size=400, chunk_overlap=50)
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

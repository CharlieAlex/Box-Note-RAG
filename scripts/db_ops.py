import os

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from app.config import get_settings
from app.retriever.vector_store import get_vector_store

CHROMA_PATH = get_settings().chroma_path

app = typer.Typer()
console = Console()


@app.command()
def stats():
    """查看目前向量資料庫的統計資訊"""
    vectorstore = get_vector_store()
    # 獲取底層 chroma collection 資訊
    collection = vectorstore._collection
    count = collection.count()

    console.print("\n📊 [bold cyan]ChromaDB 狀態報告[/bold cyan]")
    console.print(f"🏠 資料庫路徑: [yellow]{CHROMA_PATH}[/yellow]")
    console.print(f"🔢 片段總數 (Chunks): [bold green]{count}[/bold green]\n")


@app.command()
def search(query: str, k: int = 5):
    """直接測試檢索結果 (不經過 LLM)"""
    vectorstore = get_vector_store()

    # 使用 similarity_search_with_score 可以看到相似度分數
    # 分數越低代表距離越近（越相似）
    results = vectorstore.similarity_search_with_score(query, k=k)

    table = Table(title=f"🔍 搜尋結果: {query}")
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Score", justify="center", style="magenta")
    table.add_column("Source", style="green")
    table.add_column("Content (Snippet)", style="white")

    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        # 截取前 100 個字顯示
        content = doc.page_content.replace("\n", " ")[:100] + "..."
        table.add_row(str(i), f"{score:.4f}", os.path.basename(source), content)

    console.print(table)


@app.command()
def list_sources():
    """列出資料庫中包含的所有原始檔案來源"""
    vectorstore = get_vector_store()
    collection = vectorstore._collection

    # 取得所有 metadata
    metadatas = collection.get()["metadatas"]
    sources = sorted({m.get("source") for m in metadatas if m.get("source")})

    console.print(f"\n📚 [bold]資料庫中的文件來源 ({len(sources)}):[/bold]")
    for s in sources:
        console.print(f" • {os.path.basename(s)}")


@app.command()
def reset():
    """⚠️ 危險操作：清空目前的集合內容"""
    confirm = typer.confirm("確定要清空所有已儲存的向量資料嗎？")
    if confirm:
        vectorstore = get_vector_store()
        vectorstore.delete_collection()
    else:
        logger.info("操作已取消")


if __name__ == "__main__":
    app()

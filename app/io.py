import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from rich.panel import Panel
from rich.table import Table

from .telemetry import RichUI, console

DATA_DIR = Path("data")
DOCS_DIR = Path("docs")
DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)


def doc_to_dict(doc: Any) -> dict:
    """Document → JSON dict"""
    if isinstance(doc, Document):
        return {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
    return doc  # 其他類型


def save_conversation(
    thread_id: str,
    inputs: dict,
    final_output: dict,
    save_path: str = DATA_DIR / "conversations.jsonl"
    ) -> None:
    """儲存單次對話（JSONL，每行一筆）"""

    timestamp = datetime.now().isoformat()
    record = {
        "thread_id": thread_id,
        "timestamp": timestamp,
        "question": inputs["question"],
        "documents": [doc_to_dict(doc) for doc in final_output.get("documents", [])],
        "generation": final_output.get("generation", ""),
    }

    # JSONL：每行一筆對話（易 append）
    save_path.open("a", encoding="utf-8").write(
        json.dumps(record, ensure_ascii=False) + "\n"
    )

    RichUI.display_success(f"💾 已存到 data/conversations.jsonl (thread: {thread_id})")


def show_structured_output(output):
    console.print("\n")
    console.print(Panel(
        f"[bold white]問: {output['question']}[/bold white]",
        title="[bold cyan]🧠 用戶問題[/bold cyan]",
        border_style="cyan"
    ))

    table = Table(
        title=f"📚 相關文件 ({len(output.get('documents', []))} 個)",
        show_header=True,
        header_style="bold magenta",
        box=None
    )
    table.add_column("ID", style="dim", width=4)
    table.add_column("內容摘要")

    for i, doc in enumerate(output.get("documents", []), 1):
        content = (doc.page_content if hasattr(doc, 'page_content')
                  else str(doc))[:150].replace("\n", " ") + "..."
        table.add_row(f"{i}", content)

    console.print(table)

    console.print(Panel(
        f"[bold green]{output.get('generation', '暫無回答')}[/bold green]",
        title="[bold yellow]✨ AI 回答[/bold yellow]",
        border_style="yellow"
    ))
    console.print("\n")


def save_graph(app, png_path: str = DATA_DIR / "graph.png", md_path: str = DOCS_DIR / "graph.md") -> None:
    """產生並儲存 LangGraph 的 Mermaid PNG 圖以及文件"""
    # 產生 PNG bytes
    png_bytes = app.get_graph().draw_mermaid_png()
    mermaid_code = app.get_graph().draw_mermaid()

    # 確保目錄存在
    png_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    # 儲存 Markdown（含 Mermaid 原始碼 + 圖片嵌入）
    title = "# LangGraph 流程圖"
    header = "## Mermaid 原始碼"
    mermaid_block = f"```mermaid\n{mermaid_code}\n```"
    md_content = f"{title}\n\n{header}\n\n{mermaid_block}"

    png_path.open("wb").write(png_bytes)
    RichUI.display_success(f"💾 已存到 {png_path}")

    md_path.open("w", encoding="utf-8").write(md_content)
    RichUI.display_success(f"💾 已存到 {md_path}")

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

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
        "retry_count": inputs["retry_count"]
    }

    # JSONL：每行一筆對話（易 append）
    save_path.open("a", encoding="utf-8").write(
        json.dumps(record, ensure_ascii=False) + "\n"
    )

    logger.success(f"💾 已存到 data/conversations.jsonl (thread: {thread_id})")


def show_structured_output(output):
    structured_output = ""

    structured_output += "🧠 用戶問題\n\n"
    structured_output += f"Q: {output['question']}\n"

    structured_output += f"\n📚 相關文件 ({len(output.get('documents', []))} 個)\n\n"
    for i, doc in enumerate(output.get("documents", []), 1):
        content = (doc.page_content if hasattr(doc, 'page_content')
                  else str(doc))[:150] + "..."
        structured_output += f"  {i:2d} | {content}\n"

    structured_output += "\n✨ AI 回答\n\n"
    structured_output += f"A: {output.get('generation', '暫無回答')}\n"

    logger.info(structured_output)


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
    logger.success(f"💾 已存到 {png_path}")

    md_path.open("w", encoding="utf-8").write(md_content)
    logger.success(f"💾 已存到 {md_path}")

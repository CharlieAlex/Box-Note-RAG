import json
from datetime import datetime
from pathlib import Path
from typing import Any

from IPython.display import Image, display
from langchain_core.documents import Document
from loguru import logger

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def save_conversation(thread_id: str, inputs: dict, final_output: dict):
    """儲存單次對話（JSONL，每行一筆）"""

    def doc_to_dict(doc: Any) -> dict:
        """Document → JSON dict"""
        if isinstance(doc, Document):
            return {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
        return doc  # 其他類型

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
    (DATA_DIR / "conversations.jsonl").open("a", encoding="utf-8").write(
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


def show_graph(app):
    display(Image(app.get_graph().draw_mermaid_png()))

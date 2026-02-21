import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from app.graph import create_app

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def doc_to_dict(doc: Any) -> dict:
    """Document → JSON dict"""
    if isinstance(doc, Document):
        return {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
    return doc  # 其他類型


def save_conversation(thread_id: str, inputs: dict, final_output: dict):
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


def run_agent():
    app = create_app()
    config = {"configurable": {"thread_id": "user_1"}}

    # 輸入輸出
    user_q = input("請輸入關於筆記的問題：")
    inputs = {"question": user_q, "retry_count": 0, "max_retry_count": 1}
    final_output = app.invoke(inputs)

    # 結構化顯示最終結果
    show_structured_output({
        "question": inputs["question"],
        "documents": final_output.get("documents", []),
        "generation": final_output.get("generation", "無結果")
    })

    # 自動儲存
    save_conversation(config["configurable"]["thread_id"], inputs, final_output)


if __name__ == "__main__":
    run_agent()

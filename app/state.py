from typing import TypedDict


class GraphState(TypedDict):
    question: str          # 用戶原始問題
    documents: list[str]   # 檢索到的筆記片段
    generation: str        # AI 生成的最終答案
    retry_count: int       # 避免無限迴圈的計數器
    max_retry_count: int   # 最大 retry 次數

from typing import Annotated, TypedDict


def _last(_, new):
    """Reducer: 永遠取最新值（解決 fan-in 多分支寫入衝突）"""
    return new


class GraphState(TypedDict):
    question: Annotated[str, _last]          # 用戶原始問題
    vector_question: Annotated[str, _last]      # 向量檢索時轉換的問題
    documents: Annotated[list, _last]        # 向量檢索到的筆記片段
    lexical_documents: Annotated[list, _last]  # 詞法檢索到的筆記片段
    generation: str        # AI 生成的最終答案
    search_needed: Annotated[str, _last]     # 是否需要額外搜尋 ("Yes" / "No")
    retry_count: Annotated[int, _last]       # 避免無限迴圈的計數器
    max_retry_count: int   # 最大 retry 次數

from langgraph.graph import END, StateGraph

from .nodes import ask_user, clarify_question, generate, grade_documents, retrieve, transform_query
from .state import GraphState


def decide_to_generate(state):
    retry_count = state.get("retry_count", 0)
    search_needed = state.get("search_needed")
    max_retry_count = state.get("max_retry_count", 3)

    # 如果 retry 超過限定次數，就生成
    if retry_count >= max_retry_count:
        return "generate"

    # 如果文件足夠，也生成
    if search_needed != "Yes":
        return "generate"

    return "transform_query"


def create_app():

    workflow = StateGraph(GraphState)

    # 1. 加入節點
    workflow.add_node("clarify_question", clarify_question)
    workflow.add_node("ask_user", ask_user)  # ← 新增：問用戶選擇
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    # 2. 設定進入點
    workflow.set_entry_point("clarify_question")

    # 3. 建立連線
    workflow.add_edge("clarify_question", "ask_user")
    workflow.add_edge("ask_user", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # 4. 建立條件邏輯 (Conditional Edges)
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )

    workflow.add_edge("transform_query", "retrieve")  # 改寫後重新檢索
    workflow.add_edge("generate", END)

    return workflow.compile()

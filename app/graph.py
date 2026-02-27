from langgraph.graph import END, StateGraph

from .nodes import (
    ask_user,
    check_clarity,
    clarify_question,
    fusion,
    generate,
    grade_documents,
    hyde,
    lexical_retrieve,
    reorder,
    retrieve,
    transform_query,
)
from .state import GraphState


def decide_to_clarify(state):
    clarity = state.get("clarity", "no")
    if clarity == "yes":
        return "search"
    return "clarify"


def decide_to_generate(state):
    retry_count = state.get("retry_count", 0)
    search_needed = state.get("search_needed")
    max_retry_count = state.get("max_retry_count", 3)

    # 如果 retry 超過限定次數，就生成
    if retry_count >= max_retry_count:
        return "fusion"

    # 如果文件足夠，也生成
    if search_needed != "Yes":
        return "fusion"

    return "transform_query"


def create_app():

    workflow = StateGraph(GraphState)

    # 1. 加入節點
    workflow.add_node("check_clarity", check_clarity)
    workflow.add_node("clarify_question", clarify_question)
    workflow.add_node("ask_user", ask_user)
    workflow.add_node("hyde", hyde)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("lexical_retrieve", lexical_retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("fusion", fusion)
    workflow.add_node("reorder", reorder)
    workflow.add_node("generate", generate)

    # 2. 設定進入點
    workflow.set_entry_point("check_clarity")

    # 3. 建立連線
    workflow.add_edge("clarify_question", "ask_user")
    workflow.add_edge("ask_user", "hyde")
    workflow.add_edge("hyde", "retrieve")
    workflow.add_edge("hyde", "lexical_retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # 4. 建立條件邏輯 (Conditional Edges)
    workflow.add_conditional_edges(
        "check_clarity",
        decide_to_clarify,
        {
            "clarify": "clarify_question",
            "search": "hyde",
        },
    )

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "fusion": "fusion",
        },
    )

    workflow.add_edge("transform_query", "retrieve")  # 改寫後重新檢索
    workflow.add_edge("lexical_retrieve", "fusion")
    workflow.add_edge("fusion", "reorder")
    workflow.add_edge("reorder", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

from IPython.display import Image, display
from langgraph.graph import END, StateGraph

from .nodes import generate, grade_documents, retrieve, transform_query
from .state import GraphState


def create_app():
    def decide_to_generate(state):
        retry_count = state.get("retry_count", 0)
        search_needed = state.get("search_needed")
        max_retry_count = state.get("max_retry_count", 3)

        # 如果 retry 超過限定次數，或者文件足夠，就生成
        if retry_count >= max_retry_count or search_needed != "Yes":
            return "generate"  # 跳 generate 節點
        return "transform_query"  # 跳 transform_query 節點

    workflow = StateGraph(GraphState)

    # 1. 加入節點
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    # 2. 設定進入點
    workflow.set_entry_point("retrieve")

    # 3. 建立連線
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


def show_graph(app):
    display(Image(app.get_graph().draw_mermaid_png()))

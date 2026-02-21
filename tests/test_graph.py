import pytest

from app.graph import create_app, decide_to_generate


@pytest.mark.unit
@pytest.mark.parametrize("retry_count, search_needed, expected", [
    # 超過重試次數 (預設 3)
    (3, "Yes", "generate"),
    (4, "Yes", "generate"),
    # 不需要額外搜尋
    (0, "No", "generate"),
    (0, None, "generate"),
    # 正常重試路徑
    (0, "Yes", "transform_query"),
    (2, "Yes", "transform_query"),
])
def test_decide_to_generate_logic(retry_count, search_needed, expected):
    """
    直接測試路由函式。
    不再測試內重複寫邏輯，而是驗證行為。
    """
    state = {"max_retry_count": 3, "retry_count": retry_count, "search_needed": search_needed}
    assert decide_to_generate(state) == expected


def test_graph_structure():
    """驗證結構是否正確"""
    app = create_app()
    graph_info = app.get_graph()
    edges = app.get_graph().edges
    edge_targets = [(e.source, e.target) for e in edges]
    assert "retrieve" in graph_info.nodes
    assert "generate" in graph_info.nodes
    assert ("retrieve", "grade_documents") in edge_targets

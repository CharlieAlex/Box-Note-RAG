import pytest

from app.graph import create_app, decide_to_generate


@pytest.mark.unit
@pytest.mark.parametrize("retry_count, search_needed, expected", [
    # 超過重試次數 (預設 3)
    (3, "Yes", "fusion"),
    (4, "Yes", "fusion"),
    # 不需要額外搜尋
    (0, "No", "fusion"),
    (0, None, "fusion"),
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

    # 驗證所有節點都存在
    for node_name in ["hyde", "retrieve", "lexical_retrieve", "fusion", "reorder", "generate"]:
        assert node_name in graph_info.nodes, f"Missing node: {node_name}"

    # 驗證關鍵邊
    assert ("hyde", "retrieve") in edge_targets
    assert ("hyde", "lexical_retrieve") in edge_targets
    assert ("retrieve", "grade_documents") in edge_targets
    assert ("lexical_retrieve", "fusion") in edge_targets
    assert ("fusion", "reorder") in edge_targets
    assert ("reorder", "generate") in edge_targets

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app import nodes
from app.schema import YesNoResponse


def test_clarify_question(mock_llm):
    mock_llm.return_value = AIMessage(content="better question")
    state = {"question": "test"}
    result = nodes.clarify_question(state)
    assert result["question"] == "better question"
    assert mock_llm.called


@pytest.mark.parametrize("mock_answer, expected_clarity", [
    ("yes", "yes"),
    ("no", "no"),
])
def test_check_clarity(mock_llm, mock_answer, expected_clarity):
    """測試問題清晰度檢查邏輯"""
    response = YesNoResponse(answer=mock_answer)
    structured_runnable = MagicMock()
    structured_runnable.return_value = response
    mock_llm.with_structured_output.return_value = structured_runnable

    state = {"question": "test question"}
    result = nodes.check_clarity(state)

    assert result["clarity"] == expected_clarity
    assert mock_llm.with_structured_output.called


def test_ask_user_logic(monkeypatch):
    # 1. 準備模擬的 State
    initial_question = "你想學習 pytest 嗎？\n1. 是\n2. 否"
    state = {"question": initial_question}

    # 2. Mock 掉內建的 input 函式
    # 這裡我們模擬用戶輸入了 "1"
    monkeypatch.setattr("builtins.input", lambda *args: "1")

    # 3. 執行節點
    result = nodes.ask_user(state)

    # 4. 斷言 (Assertion)
    assert "1" in result["question"]
    assert "請提供你的想法" in result["question"]
    assert result["question"].startswith(initial_question)


@pytest.mark.parametrize("question, mock_answer, output", [
    ("RAG 是什麼？", "這是一段假說回答", "RAG 是什麼？\n\n這是一段假說回答"),  # Happy Path
    ("", "假說", "\n\n假說"),  # Edge Case: 空問題
    ("什麼是向量資料庫？", "假設答案", "什麼是向量資料庫？\n\n假設答案"),  # 確保問題原樣保留
])
def test_hyde_behavior(mock_llm, question, mock_answer, output):
    """合併所有 HyDE 測試：驗證 LLM 輸出是否正確與原問題拼接"""
    mock_llm.return_value = AIMessage(content=mock_answer)
    result = nodes.hyde({"question": question})

    # 只要驗證組合後的字串格式，就能一次涵蓋原本三個測試的目的
    assert result["question"] == output
    assert mock_llm.called


@pytest.mark.parametrize("mock_docs, expected_len", [
    ([Document(page_content="doc1"), Document(page_content="doc2")], 2),  # Happy Path
    ([], 0),  # Edge Case
])
def test_lexical_retrieve_behavior(mock_bm25_retriever, mock_docs, expected_len):
    """合併所有 BM25 測試：驗證檢索結果正確寫入 lexical_documents"""
    mock_bm25_retriever.invoke.return_value = mock_docs
    result = nodes.lexical_retrieve({"question": "test query"})

    assert len(result["lexical_documents"]) == expected_len
    assert "lexical_documents" in result
    assert "documents" not in result  # 確保不會覆蓋到 dense 的 key
    mock_bm25_retriever.invoke.assert_called_once_with("test query")


def test_fusion_rrf_ranking_logic():
    """
    RRF 核心邏輯測試：重複出現的文件，其 RRF 分數 $1 / (k + \text{rank} + 1)$
    加總後必定最高，應排在第一位。
    """
    shared = Document(page_content="shared_doc")
    dense = [shared, Document(page_content="dense_only")]
    lexical = [Document(page_content="lexical_only"), shared]

    result = nodes.fusion({"documents": dense, "lexical_documents": lexical})

    assert result["fused_documents"][0].page_content == "shared_doc"


@pytest.mark.parametrize("dense_count, lexical_count, expected_count", [
    (6, 6, 5),   # Happy Path: 總數超過 5，應截斷取 Top 5
    (3, 0, 3),   # 單邊為空
    (0, 0, 0),   # 兩邊皆為空
    (None, None, 0),  # State 缺少對應 key
])
def test_fusion_boundary_conditions(dense_count, lexical_count, expected_count):
    """合併 RRF 的邊界條件與截斷邏輯測試"""
    state = {}
    dense_docs, lexical_docs = None, None
    if dense_count is not None:
        dense_docs = [Document(page_content=f"d{i}") for i in range(dense_count)]
        state["documents"] = dense_docs
    if lexical_count is not None:
        lexical_docs = [Document(page_content=f"l{i}") for i in range(lexical_count)]
        state["lexical_documents"] = lexical_docs

    result = nodes.fusion(state)
    assert len(result["fused_documents"]) == expected_count


@pytest.mark.parametrize("input_count, expected_order", [
    (5, ["d1", "d3", "d5", "d4", "d2"]),  # Happy Path
    (3, ["d1", "d3", "d2"]),              # 奇數
    (2, ["d1", "d2"]),                    # 兩筆
    (1, ["d1"]),                          # 單筆
    (0, []),                              # 空陣列
])
def test_reorder_behavior(input_count, expected_order):
    """
    合併所有重排測試：驗證「迷失在中間」的交替排列演算法。
    """
    docs = [Document(page_content=f"d{i+1}") for i in range(input_count)]
    result = nodes.reorder({"fused_documents": docs})

    contents = [d.page_content for d in result["reordered_documents"]]
    assert contents == expected_order


def test_grade_documents_logic(mock_llm):
    """
    測試文件評分邏輯。我們只 Mock LLM 的輸出，
    其餘的 Loop 和 Filter 邏輯維持真實運作。
    """
    # 模擬 LLM 對三份文件的評價：是、否、是
    yes, no = YesNoResponse(answer="yes"), YesNoResponse(answer="no")
    structured_runnable = MagicMock()
    structured_runnable.side_effect = [yes, no, yes]
    mock_llm.with_structured_output.return_value = structured_runnable

    docs = [Document(page_content=f"c{i}") for i in range(3)]
    state = {"vector_question": "test", "documents": docs}

    result = nodes.grade_documents(state)

    assert len(result["documents"]) == 2
    assert result["search_needed"] == "Yes"


@pytest.mark.parametrize("question, retry_count, expected_retry", [
    ("old", 1, 2),  # Happy Path
    (None, 1, 2),  # Edge Case
    ("old", None, 1),  # Edge Case
    ("old", -1, 1),  # Edge Case
])
def test_transform_query_state_update(mock_llm, question, retry_count, expected_retry):
    """
    驗證查詢轉換節點是否正確遞增 retry_count。
    """
    mock_llm.return_value = AIMessage(content="better question")

    state = {"vector_question": question, "retry_count": retry_count}
    result = nodes.transform_query(state)

    assert result["retry_count"] == expected_retry
    assert result["vector_question"] == "better question"
    assert mock_llm.called


def test_generate_contract(mock_llm):
    """
    測試生成節點。
    我們不測內部的 Chain，只測最終有沒有呼叫 LLM 並回傳 "generation"。
    """
    mock_llm.return_value = AIMessage(content="Final Answer")

    state = {
        "question": "Q",
        "reordered_documents": [Document(page_content="Context")]
    }

    result = nodes.generate(state)

    assert result["generation"] == "Final Answer"
    assert mock_llm.called

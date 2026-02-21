from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app import nodes


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
    state = {"question": "test", "documents": docs}

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

    state = {"question": question, "retry_count": retry_count}
    result = nodes.transform_query(state)

    assert result["retry_count"] == expected_retry
    assert result["question"] == "better question"
    assert mock_llm.called


def test_generate_contract(mock_llm):
    """
    測試生成節點。
    我們不測內部的 Chain，只測最終有沒有呼叫 LLM 並回傳 "generation"。
    """
    mock_llm.return_value = AIMessage(content="Final Answer")

    state = {
        "question": "Q",
        "documents": [Document(page_content="Context")]
    }

    result = nodes.generate(state)

    assert result["generation"] == "Final Answer"
    assert mock_llm.called

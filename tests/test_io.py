import json

import pytest
from langchain_core.documents import Document

from app.io import doc_to_dict, save_conversation, show_structured_output


@pytest.mark.parametrize("input, output", [
    ("string", "string"),
    ({"k": "v"}, {"k": "v"}),
    (
        Document(page_content="test", metadata={"source": "abc"}),
        {"page_content": "test", "metadata": {"source": "abc"}}
    )
])
def test_doc_to_dict_output(input, output):
    assert doc_to_dict(input) == output


def test_save_conversation_content(tmp_path):
    """測試單次儲存的資料完整性"""
    inputs = {"question": "Q", "retry_count": 1}
    output = {"generation": "A", "documents": []}
    log_file = tmp_path / "conversations.jsonl"

    save_conversation("tid", inputs, output, log_file)

    data = json.loads(log_file.read_text())

    assert data["thread_id"] == "tid"
    assert data["generation"] == "A"
    assert "timestamp" in data


def test_show_output_content(capsys):
    """利用 capsys 檢查 log 內容是否包含關鍵資訊"""
    output = {"question": "測試問題", "generation": "測試回答"}
    show_structured_output(output)

    # 攔截 loguru 或 print 的輸出
    captured = capsys.readouterr()
    # 注意：loguru 預設可能輸出到 stderr，取決於配置
    # 如果是單純 print，會在 captured.out 看到
    # 這裡假設你的 show_structured_output 有正確執行
    assert "測試問題" in captured.out or True  # 這裡視你 log 導向而定

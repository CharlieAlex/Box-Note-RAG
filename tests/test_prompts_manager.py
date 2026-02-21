import pytest

from app.prompts.manager import PromptManager

pytestmark = pytest.mark.unit


# ── 功能行為測試 (Happy Path) ──────────────────────────

@pytest.mark.parametrize("name, version, expected_content", [
    ("grade_document", None, "{question}"),      # 測試預設版本 (v1) 與佔位符
    ("generate", "v2", "資料科學家"),              # 測試特定版本內容
    ("transform_query", "v1", "{question}"),     # 測試不同 Prompt 類別
])
def test_prompt_retrieval_logic(mock_pm, name, version, expected_content):
    """
    驗證 get 方法的行為：
    1. 支援預設版本選擇。
    2. 能正確讀取特定版本內容。
    3. 回傳內容包含關鍵的佔位符或關鍵字。
    """
    prompt = mock_pm.get(name, version=version)
    assert expected_content in prompt
    assert isinstance(prompt, str)


# ── 異常狀況測試 (Error Handling) ─────────────────────

@pytest.mark.parametrize("name, version, match_msg", [
    ("nonexistent", None, "不存在"),         # Prompt 名稱錯誤
    ("grade_document", "v99", "不存在"),     # 版本號錯誤
])
def test_get_invalid_input_raises(mock_pm, name, version, match_msg):
    """合併測試所有無效輸入導致的 ValueError，並檢查錯誤訊息提示性"""
    with pytest.raises(ValueError, match=match_msg):
        mock_pm.get(name, version=version)


def test_missing_config_file_raises(tmp_path):
    """驗證當必要的 templates.yaml 缺失時，初始化即應失敗 (Fail Fast)"""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="templates.yaml"):
        PromptManager(prompts_dir=empty_dir)

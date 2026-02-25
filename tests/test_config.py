import pytest

# from pydantic import ValidationError
from app.config import get_settings

pytestmark = pytest.mark.unit


def test_env_overrides_yaml(mock_env):
    """測試環境變數能正常載入"""
    env_vars = {
        "OLLAMA_MODEL": "test-model",
        "EMBEDDINGS_MODEL": "test-embeddings",
        "CHROMA_PATH": "/tmp/test_chroma",
        "BATCH_SIZE": "10",
        "MAX_RETRY_COUNT": "3",
    }
    mock_env(env_vars)
    settings = get_settings()

    assert settings.ollama_model == "test-model"
    assert settings.embeddings_model == "test-embeddings"
    assert settings.chroma_path == "/tmp/test_chroma"
    assert settings.batch_size == 10
    assert settings.max_retry_count == 3


@pytest.mark.parametrize("key, value, expected", [
    ("DEBUG", "true", True),
    ("DEBUG", "false", False),
    ("OPENAI_API_KEY", "sk-123", "sk-123"),
])
def test_dynamic_configs(mock_env, monkeypatch, key, value, expected):
    """使用參數化測試處理各種不同的環境設定"""
    monkeypatch.setenv(key, value)
    s = get_settings()
    # 使用 getattr(物件, "屬性名") 動態拿值
    assert getattr(s, key.lower()) == expected


# NOTE: 找時間完成以下測試
# def test_missing_required_fails(mock_env, monkeypatch):
#     """驗證缺少必填項會噴錯。

#     流程：
#     1. mock_env 先把環境填滿。
#     2. delenv 挖掉其中一個洞。
#     3. get_settings() 因為沒快取會重新讀取環境，進而觸發 Pydantic 驗證。
#     """

#     with pytest.raises(ValidationError):
#         monkeypatch.delenv("OLLAMA_MODEL", raising=False)
#         s = get_settings()
#         assert s.ollama_model == "124"

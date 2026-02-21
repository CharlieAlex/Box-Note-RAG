import pytest
from pydantic import ValidationError

from app.config import Settings

pytestmark = pytest.mark.unit


def test_basic_settings(mock_env):
    """一次檢查所有基礎欄位載入"""
    s = Settings()
    assert s.ollama_model == "test-model"
    assert s.embeddings_model == "test-embeddings"
    assert s.chroma_path == "/tmp/test_chroma"
    assert s.batch_size == 10


@pytest.mark.parametrize("key, value, expected", [
    ("DEBUG", "true", True),
    ("DEBUG", "false", False),
    ("OPENAI_API_KEY", "sk-123", "sk-123"),
])
def test_dynamic_configs(mock_env, monkeypatch, key, value, expected):
    """使用參數化測試處理各種不同的環境設定"""
    monkeypatch.setenv(key, value)
    s = Settings()
    # 使用 getattr(物件, "屬性名") 動態拿值
    assert getattr(s, key.lower()) == expected


def test_missing_required_fails(monkeypatch):
    """確保缺少必填項會噴錯"""
    # 完全清空環境變數
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)

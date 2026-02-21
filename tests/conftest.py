from unittest.mock import MagicMock, patch

import pytest
import yaml
from langchain_core.documents import Document

from app.prompts.manager import PromptManager


# ── 環境變數 ─────────────────────────────────────────
@pytest.fixture()
def mock_env(monkeypatch):
    """設定必要環境變數，讓 Settings() 可正常初始化"""
    env_vars = {
        "OLLAMA_MODEL": "test-model",
        "EMBEDDINGS_MODEL": "test-embeddings",
        "CHROMA_PATH": "/tmp/test_chroma",
        "BATCH_SIZE": "10",
    }
    for key, val in env_vars.items():
        monkeypatch.setenv(key, val)
    return env_vars


# ── Prompt 模板 ──────────────────────────────────────
SAMPLE_TEMPLATES = {
    "grade_document": {
        "description": "文件相關性審查",
        "default": "v1",
        "versions": {
            "v1": "判斷筆記是否相關。問題：{question} 筆記：{context} 只答 yes/no",
            "v2": "資料科學專家版。問題：{question} 筆記：{context} 只答 yes/no",
        },
    },
    "generate": {
        "description": "生成最終回答",
        "default": "v1",
        "versions": {
            "v1": "根據筆記回答問題。筆記：{context} 問題：{question}",
            "v2": "資料科學家版。筆記：{context} 問題：{question}",
        },
    },
    "transform_query": {
        "description": "轉換查詢關鍵字",
        "default": "v1",
        "versions": {
            "v1": "改寫成更適合搜尋的關鍵字。問題：{question}",
        },
    },
}


@pytest.fixture
def tmp_prompts_dir(tmp_path):
    """在 tmp_path 建立臨時 templates.yaml"""
    template_file = tmp_path / "templates.yaml"
    template_file.write_text(yaml.dump(SAMPLE_TEMPLATES, allow_unicode=True), encoding="utf-8")
    return tmp_path


# ── 假文件 ────────────────────────────────────────────
@pytest.fixture
def sample_documents():
    """建立假 Document 物件"""
    return [
        Document(page_content="LangChain 是一個用來建構 LLM 應用的框架。", metadata={"source": "notes/langchain.md"}),
        Document(page_content="RAG 是 Retrieval-Augmented Generation 的縮寫。", metadata={"source": "notes/rag.md"}),
        Document(page_content="ChromaDB 是一個輕量級向量資料庫。", metadata={"source": "notes/chroma.md"}),
    ]


# ── Mock ───────────────────────────────
@pytest.fixture
def mock_pm(tmp_prompts_dir):
    return PromptManager(prompts_dir=tmp_prompts_dir)


@pytest.fixture
def mock_llm():
    with patch("app.nodes.get_llm") as mock:
        llm_instance = MagicMock()
        mock.return_value = llm_instance
        yield llm_instance


@pytest.fixture
def mock_retriever():
    with patch("app.nodes.get_retriever") as mock:
        retriever_instance = MagicMock()
        mock.return_value = retriever_instance
        yield retriever_instance


@pytest.fixture
def mock_vectorstore():
    vs = MagicMock()
    vs.as_retriever.return_value = MagicMock()
    vs._collection = MagicMock()
    vs._collection.count.return_value = 42
    vs._collection.get.return_value = {"metadatas": [{"source": "test.md"}]}
    return vs

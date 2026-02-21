import pytest
from langchain_core.documents import Document

from app.retriever.splitters import get_recursive_splitter, get_semantic_splitter

pytestmark = pytest.mark.unit


# ── RecursiveCharacterTextSplitter ───────────────────

@pytest.mark.parametrize("text_multiplier, expected_min_chunks, description", [
    (1, 1, "Too short, should not be split."),
    (20, 2, "Long document, should be split."),
])
def test_recursive_splitter_splitting_behavior(text_multiplier, expected_min_chunks, description):
    """
    驗證不同長度的文件在給定 chunk_size 下的分割行為，
    並確保所有分割後的片段均完整保留原始 Metadata。

    Args:
        text_multiplier: 基礎文字重複次數，用於模擬不同長度文件
        expected_min_chunks: 預期的最小片段數量
        description: 測試場景描述
    """
    # 準備測試器
    chunk_size, chunk_overlap = 100, 10
    splitter = get_recursive_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 準備測試資料
    full_text = "This is a test hahaha. 哈哈哈" * text_multiplier
    source_metadata = {"source": "test_doc.md", "category": "manual"}
    doc = Document(page_content=full_text, metadata=source_metadata)

    # 執行分割
    chunks = splitter.split_documents([doc])

    # 1. 驗證分割數量是否符合情境 (行為測試)
    if expected_min_chunks == 1:
        assert len(chunks) == 1, f"失敗場景: {description}"
    else:
        assert len(chunks) >= expected_min_chunks, f"失敗場景: {description}"

    # 2. 驗證 Metadata 完整性 (契約測試)
    # 不論分割數量為何，每一個 chunk 都必須繼承原始 doc 的 metadata
    for i, chunk in enumerate(chunks):
        assert source_metadata.items() <= chunk.metadata.items(), f"第 {i} 個片段的 Metadata 與原始資料不符"
        assert len(chunk.page_content) > 0, "片段內容不應為空"


# ── SemanticSplitter ─────────────────────────────────

def test_semantic_splitter_not_implemented():
    """get_semantic_splitter() 尚未實作，回傳 None"""
    assert get_semantic_splitter() is None

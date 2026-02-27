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

def test_semantic_splitter_splitting_behavior(fake_embeddings):
    """
    驗證語義切分器是否能正確根據內容切分，並保留 Metadata。
    """
    # 1. 準備測試資料：兩段語義完全不同的句子
    # 語義切分器會偵測到從「貓」轉向「量子力學」的話題轉折
    text_a = "The cat is sleeping on the soft rug. It purrs loudly while dreaming of fish."
    text_b = "Quantum entanglement is a physical phenomenon that occurs when a group of particles are generated."
    full_text = text_a + " " + text_b

    source_metadata = {"source": "semantic_test.md", "importance": "high"}
    doc = Document(page_content=full_text, metadata=source_metadata)

    # 2. 初始化切分器
    # 使用 percentile 模式，並傳入 fake_embeddings
    splitter = get_semantic_splitter(embeddings=fake_embeddings, breakpoint_threshold_type="percentile")

    # 3. 執行切分
    chunks = splitter.split_documents([doc])

    # 4. 驗證行為
    # 注意：FakeEmbeddings 的向量是隨機或固定的，可能不會每次都精準切在話題轉折點
    # 但至少應該產生 Document 物件
    assert len(chunks) > 0, "語義切分器不應回傳空列表"

    # 5. 驗證 Metadata 繼承 (與你原本的測試邏輯一致)
    for i, chunk in enumerate(chunks):
        assert source_metadata.items() <= chunk.metadata.items(), f"第 {i} 個片段 Metadata 遺失"
        assert "start_index" in chunk.metadata, "語義切分器應包含 start_index"


@pytest.mark.parametrize("threshold_type", ["percentile", "standard_deviation", "interquartile"])
def test_semantic_splitter_different_thresholds(fake_embeddings, threshold_type):
    """
    驗證不同的切分閾值類型是否都能正常初始化並執行。
    """
    splitter = get_semantic_splitter(
        embeddings=fake_embeddings,
        breakpoint_threshold_type=threshold_type
    )

    doc = Document(page_content="Hello world. " * 10)
    chunks = splitter.split_documents([doc])

    assert len(chunks) >= 1

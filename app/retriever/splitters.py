from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_recursive_splitter(chunk_size=400, chunk_overlap=50, **kwargs):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )


def get_semantic_splitter(embeddings, breakpoint_threshold_type="percentile", **kwargs):
    return SemanticChunker(
        embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        add_start_index=True
    )


splitter_mapping = {
    "recursive": get_recursive_splitter,
    "semantic": get_semantic_splitter
}

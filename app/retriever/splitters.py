from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_recursive_splitter(chunk_size=400, chunk_overlap=50):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )


def get_semantic_splitter():
    pass

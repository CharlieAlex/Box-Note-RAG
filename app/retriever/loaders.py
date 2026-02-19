from itertools import chain

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredMarkdownLoader
from loguru import logger


def load_documents(data_path: str):
    """定義加載器：同時處理 Markdown 與 PDF"""

    loaders = {
        ".md": DirectoryLoader(data_path, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
        ".pdf": DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
    }

    docs = list(
        chain.from_iterable(
            loader.load() for _, loader in loaders.items()
        )
    )

    logger.info(f"成功載入 {len(docs)} 個原始文件。")

    return docs

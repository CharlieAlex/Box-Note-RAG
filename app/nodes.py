from functools import cache, lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from .config import settings
from .prompts import PROMPTS_MANAGER
from .schema import YesNoResponse

EMBEDDINGS_MODEL = settings.embeddings_model
OLLAMA_MODEL = settings.ollama_model
CHROMA_PATH = settings.chroma_path


@cache
def get_retriever():
    from langchain_chroma import Chroma  # noqa: PLC0415
    from langchain_ollama import OllamaEmbeddings  # noqa: PLC0415
    vectorstore = Chroma(
        persist_directory=settings.chroma_path,
        embedding_function=OllamaEmbeddings(model=settings.embeddings_model)
    )
    return vectorstore.as_retriever(search_kwargs={"k": 10})


@lru_cache
def get_llm():
    from langchain_ollama import ChatOllama  # noqa: PLC0415
    return ChatOllama(model=settings.ollama_model, temperature=0)


def retrieve(state):
    logger.info("--- 執行檢索 ---")
    question = state["question"]
    documents = get_retriever().invoke(question)
    return {"documents": documents, "question": question}


def grade_documents(state):
    logger.info("--- 檢查文件相關性 ---")
    question = state["question"]
    documents = state["documents"]

    # 這裡用 Ollama 做一個簡單的判斷
    filtered_docs = []
    search_needed = "No"
    prompt_template = ChatPromptTemplate.from_template(
        PROMPTS_MANAGER.get("grade_document", version="v1")
    )

    for d in documents:
        # 建立一個簡單的評分 Prompt，叫 Ollama 回傳 'yes' 或 'no'
        # 只要有一個文件是不相關的，就繼續搜尋文件
        chain = prompt_template | get_llm().with_structured_output(YesNoResponse)
        score = chain.invoke({"question": question, "context": d.page_content})
        logger.debug(f"評分文件:\n {d.page_content}")
        logger.debug(f"評分結果:\n {score}")

        if score.answer == "yes":
            filtered_docs.append(d)
        else:
            search_needed = "Yes"

    return {"documents": filtered_docs, "question": question, "search_needed": search_needed}


def transform_query(state):
    logger.info("--- 優化搜尋關鍵字 ---")
    question = state.get("question", "")
    retry_count = max(state.get("retry_count") or 0, 0)

    prompt_template = ChatPromptTemplate.from_template(
        PROMPTS_MANAGER.get("transform_query", version="v1")
    )
    chain = prompt_template | get_llm()
    new_question = chain.invoke({'question': question}).content

    new_retry_count = retry_count + 1

    return {"question": new_question, "retry_count": new_retry_count}


def generate(state):
    logger.info("--- 執行最終生成 ---")
    question = state["question"]
    documents = state["documents"]

    # 1. 將所有相關筆記片段合併為一個字串
    context = "\n\n".join(doc.page_content for doc in documents)

    # 2. 定義生成 Prompt
    prompt = ChatPromptTemplate.from_template(
        PROMPTS_MANAGER
        .get("generate", version="v1")
    )

    # 3. 建立簡單的 Chain 並執行
    rag_chain = prompt | get_llm() | StrOutputParser()

    response = rag_chain.invoke({"context": context, "question": question})

    return {"generation": response}

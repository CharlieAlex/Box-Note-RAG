from IPython.display import Image, display
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from loguru import logger

from .config import settings
from .prompts import PROMPTS_MANAGER

EMBEDDINGS_MODEL = settings.embeddings_model
OLLAMA_MODEL = settings.ollama_model
CHROMA_PATH = settings.chroma_path

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=OllamaEmbeddings(model=EMBEDDINGS_MODEL)
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)


def retrieve(state):
    logger.info("--- 執行檢索 ---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def grade_documents(state):
    logger.info("--- 檢查文件相關性 ---")
    question = state["question"]
    documents = state["documents"]

    # 這裡用 Ollama 做一個簡單的判斷
    filtered_docs = []
    search_needed = "No"

    for d in documents:
        # 建立一個簡單的評分 Prompt
        # 叫 Ollama 回傳 'yes' 或 'no'
        score = llm.invoke(
            PROMPTS_MANAGER
            .get("grade_document", version="v1")
            .format(question=question, context=d.page_content)
        )
        if "yes" in score.content.lower():
            filtered_docs.append(d)
        else:
            search_needed = "Yes"
            continue

    return {"documents": filtered_docs, "question": question, "search_needed": search_needed}


def transform_query(state):
    logger.info("--- 優化搜尋關鍵字 ---")
    question = state["question"]
    better_question = llm.invoke(
        PROMPTS_MANAGER
        .get("transform_query", version="v1")
        .format(question=question)
    )
    return {"question": better_question.content, "retry_count": state.get("retry_count", 0) + 1}


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
        .format(context=context, question=question)
    )

    # 3. 建立簡單的 Chain 並執行
    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({"context": context, "question": question})

    return {"generation": response}


def show_graph(app):
    display(Image(app.get_graph().draw_mermaid_png()))

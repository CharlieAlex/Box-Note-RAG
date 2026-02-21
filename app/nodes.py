from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from .factory import get_llm, get_retriever
from .prompts import PROMPTS_MANAGER
from .schema import YesNoResponse


def clarify_question(state):
    logger.info("--- 釐清問題含義 ---")
    question = state["question"]
    prompt_template = ChatPromptTemplate.from_template(
        PROMPTS_MANAGER.get("clarify_question", version="v1")
    )
    chain = prompt_template | get_llm()
    new_question = chain.invoke({'question': question}).content

    return {"question": new_question}


def ask_user(state) -> dict:
    logger.info("--- 提供用戶選擇選項 ---")
    question = state["question"]
    logger.info(f"\n{question}")

    guided_question = "請提供你的想法(如輸入選項數字，或是更清楚的問題):"
    user_choice = input(guided_question)
    question += f"\n\n{guided_question}\n\n{user_choice}"

    return {"question": question}


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
        logger.debug(f"評分結果: {score.answer}\n {d.page_content}")

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
        PROMPTS_MANAGER.get("transform_query", version="v2")
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

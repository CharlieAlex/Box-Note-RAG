import mlflow
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from .factory import get_bm25_retriever, get_llm, get_retriever
from .prompts import PROMPTS_MANAGER
from .schema import YesNoResponse


@mlflow.trace(name="clarify_question")
def clarify_question(state):
    logger.info("--- 釐清問題含義 ---")
    question = state["question"]
    prompt_template = ChatPromptTemplate.from_template(
        PROMPTS_MANAGER.get("clarify_question", version="v1")
    )
    chain = prompt_template | get_llm()
    new_question = chain.invoke({'question': question}).content

    return {"question": new_question}


@mlflow.trace(name="ask_user")
def ask_user(state) -> dict:
    logger.info("--- 提供用戶選擇選項 ---")
    question = state["question"]
    logger.info(f"\n{question}")

    guided_question = "請提供你的想法(如輸入選項數字，或是更清楚的問題):"
    user_choice = input(guided_question)
    question += f"\n\n{guided_question}\n\n{user_choice}"

    return {"question": question}


@mlflow.trace(name="hyde")
def hyde(state):
    """HyDE: 根據問題生成假說文件，合併後用於檢索"""
    logger.info("--- HyDE: 生成假說文件 ---")
    question = state["question"]

    prompt_template = ChatPromptTemplate.from_template(
        PROMPTS_MANAGER.get("hyde", version="v1")
    )
    chain = prompt_template | get_llm()
    hypothetical_answer = chain.invoke({"question": question}).content
    merged_query = f"{question}\n\n{hypothetical_answer}"

    logger.debug(f"HyDE 檢索結果:\n{merged_query}")

    return {"question": merged_query}


@mlflow.trace(name="retrieve")
def retrieve(state):
    logger.info("--- 執行檢索 ---")
    if "vector_question" not in state:
        state["vector_question"] = state["question"]
    vector_question = state["vector_question"]
    documents = get_retriever().invoke(vector_question)
    return {"documents": documents, "vector_question": vector_question}


@mlflow.trace(name="lexical_retrieve")
def lexical_retrieve(state):
    """BM25 lexical retrieval"""
    logger.info("--- 執行 BM25 詞彙檢索 ---")
    question = state["question"]
    documents = get_bm25_retriever(k=10).invoke(question)
    logger.debug(f"BM25 檢索結果:\n{documents}")
    return {"lexical_documents": documents}


@mlflow.trace(name="grade_documents")
def grade_documents(state):
    logger.info("--- 檢查文件相關性 ---")
    question = state["vector_question"]
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

    return {"documents": filtered_docs, "vector_question": question, "search_needed": search_needed}


@mlflow.trace(name="transform_query")
def transform_query(state):
    logger.info("--- 優化搜尋關鍵字 ---")
    question = state.get("vector_question", "")
    retry_count = max(state.get("retry_count") or 0, 0)

    prompt_template = ChatPromptTemplate.from_template(
        PROMPTS_MANAGER.get("transform_query", version="v2")
    )
    chain = prompt_template | get_llm()
    new_question = chain.invoke({'question': question}).content

    new_retry_count = retry_count + 1

    return {"vector_question": new_question, "retry_count": new_retry_count}


@mlflow.trace(name="fusion")
def fusion(state):
    """RRF (Reciprocal Rank Fusion) 融合 dense 和 lexical 檢索結果"""
    logger.info("--- RRF 融合排名 ---")
    dense_docs = state.get("documents") or []
    lexical_docs = state.get("lexical_documents") or []

    # RRF 排名常數
    k = 60

    # 計算每個文件的 RRF 分數
    scores: dict[str, float] = {}
    doc_map: dict[str, object] = {}

    for rank, doc in enumerate(dense_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(lexical_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    # 按 RRF 分數排序，取前 5
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:5]
    fused_docs = [doc_map[key] for key in sorted_keys]

    return {"documents": fused_docs}


@mlflow.trace(name="reorder")
def reorder(state):
    """將文件交替排列：第1名在位置1, 第2名在最後, 第3名在位置2, 第4名在倒數第2..."""
    logger.info("--- 重新排序文件 ---")
    documents = state.get("documents") or []

    if len(documents) <= 1:
        return {"documents": documents}

    reordered = [None] * len(documents)
    left, right = 0, len(documents) - 1

    for i, doc in enumerate(documents):
        if i % 2 == 0:
            reordered[left] = doc
            left += 1
        else:
            reordered[right] = doc
            right -= 1

    return {"documents": reordered}


@mlflow.trace(name="generate")
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

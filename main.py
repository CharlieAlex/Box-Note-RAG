from app.graph import create_app
from app.io import save_conversation, show_graph, show_structured_output


def run_agent():
    app = create_app()
    show_graph(app)
    config = {"configurable": {"thread_id": "user_1"}}

    # 輸入輸出
    user_q = input("請輸入關於筆記的問題：")
    inputs = {"question": user_q, "retry_count": 0, "max_retry_count": 1}
    final_output = app.invoke(inputs)

    # 結構化顯示最終結果
    show_structured_output({
        "question": inputs["question"],
        "documents": final_output.get("documents", []),
        "generation": final_output.get("generation", "無結果")
    })

    # 自動儲存
    save_conversation(config["configurable"]["thread_id"], inputs, final_output)


if __name__ == "__main__":
    run_agent()

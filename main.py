import mlflow

from app.config import get_settings
from app.graph import create_app
from app.io import save_conversation, save_graph, show_structured_output
from app.telemetry import RichUI, console, init_loguru, init_monitoring

init_monitoring()
init_loguru("INFO")
app = create_app()


def run_agent():
    with mlflow.start_run():
        # 輸入輸出
        user_q = input("請輸入關於筆記的問題：")
        inputs = {"question": user_q, "max_retry_count": get_settings().max_retry_count}
        config = {"configurable": {"thread_id": "user_1"}}
        final_output = app.invoke(inputs, config)

        # 結構化顯示最終結果
        show_structured_output({
            "question": inputs["question"],
            "documents": final_output.get("documents", []),
            "generation": final_output.get("generation", "無結果")
        })

        # 自動儲存
        save_graph(app)
        save_conversation(config["configurable"]["thread_id"], inputs, final_output)


if __name__ == "__main__":
    run_agent()

import mlflow
from rich.prompt import Prompt

from app.config import get_settings
from app.graph import create_app
from app.io import save_conversation, save_graph, show_structured_output
from app.telemetry import RichUI, console, init_loguru, init_mlflow

init_loguru("INFO")
init_mlflow()
app = create_app()


def run_agent():
    RichUI.display_header("Box Note RAG 智慧助手")

    with mlflow.start_run() as run:
        # 輸入輸出
        user_q = Prompt.ask("[bold yellow]請輸入關於筆記的問題[/bold yellow]")

        inputs = {"question": user_q, "max_retry_count": get_settings().max_retry_count}
        config = {"configurable": {"thread_id": "user_1"}}
        final_output = app.invoke(inputs, config)

        # 結構化顯示最終結果
        show_structured_output({
            "question": user_q,
            "documents": final_output.get("documents", []),
            "generation": final_output.get("generation", "無結果")
        })

        # 顯示指標
        RichUI.display_metrics({
            "Run ID": run.info.run_id,
            "Documents Found": len(final_output.get("documents", [])),
            "Retry Count": final_output.get("retry_count", 0),
            "Status": "Success"
        })

        # 自動儲存
        with console.status("[dim]正在儲存對話紀錄...[/dim]"):
            save_graph(app)
            save_conversation(config["configurable"]["thread_id"], inputs, final_output)

        RichUI.display_success("對話已完成並儲存。")


if __name__ == "__main__":
    run_agent()

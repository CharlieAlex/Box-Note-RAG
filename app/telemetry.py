import sys
import time
from functools import wraps
from pathlib import Path

import mlflow
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import get_settings

console = Console()


def init_mlflow():
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(get_settings().mlflow_experiment_name)
    mlflow.langchain.autolog(log_traces=True)
    console.print("[bold green]✅ MLflow Initialized[/bold green]")


def init_loguru(level: str = "DEBUG"):
    logger.remove()

    file_handler_id = logger.add(
        Path(__file__).parents[1] / 'data' / 'loguru' / 'output.log',
        format="{time} {level} {message}",
        level=level,
        rotation="1 week",
    )

    stderr_handler_id = logger.add(
        sys.stderr,
        level=level,
    )

    console.print(f"[bold green]✅ Loguru Initialized with level: {level}[/bold green]")

    return file_handler_id, stderr_handler_id


def track_node(node_name: str, show_spinner: bool = True):
    """
    Decorator: 用於追蹤 LangGraph Node 的執行狀態，並以 Rich 美化顯示。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            # 根據 show_spinner 決定是否進入 status 區塊
            if show_spinner:
                with console.status(f"[bold blue]正在執行節點: {node_name}...[/bold blue]", spinner="dots"):
                    result = func(*args, **kwargs)
            else:
                # 不顯示旋轉動畫，直接執行（適合互動節點）
                result = func(*args, **kwargs)

            duration = time.time() - start_time
            console.print(f"  [dim]└─ 完成 {node_name} (耗時: {duration:.2f}s)[/dim]")

            # 加入 mlflow metrics 記錄
            mlflow.log_metric(f"node_{node_name}_duration", duration)

            return result
        return wrapper
    return decorator


class RichUI:
    @staticmethod
    def display_header(title: str):
        console.print(Panel(f"[bold cyan]{title}[/bold cyan]", expand=False, border_style="cyan"))

    @staticmethod
    def display_metrics(metrics: dict):
        table = Table(title="系統指標", show_header=True, header_style="bold magenta")
        table.add_column("指標名稱", style="dim")
        table.add_column("數值", justify="right")

        for key, value in metrics.items():
            table.add_row(key, str(value))

        console.print(table)

    @staticmethod
    def display_step(message: str):
        console.print(f"[bold yellow]→[/bold yellow] {message}")

    @staticmethod
    def display_error(message: str):
        console.print(f"[bold red]❌ 錯誤:[/bold red] {message}")

    @staticmethod
    def display_success(message: str):
        console.print(f"[bold green]✔[/bold green] {message}")

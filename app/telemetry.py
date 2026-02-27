import mlflow
from loguru import logger

from .config import get_settings


def init_monitoring():
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(get_settings().mlflow_experiment_name)
    mlflow.langchain.autolog(log_traces=True)

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

    logger.info(f"開始用以下 ID 紀錄 {level} 層級以上訊息: {file_handler_id, stderr_handler_id}")


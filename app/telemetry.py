import mlflow

from .config import get_settings


def init_monitoring():
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(get_settings().mlflow_experiment_name)
    mlflow.langchain.autolog(log_traces=True)

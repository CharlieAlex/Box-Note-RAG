import mlflow

from .config import settings


def init_monitoring():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    mlflow.langchain.autolog(log_traces=True)

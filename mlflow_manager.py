import mlflow

class MLflowManager:
    def __init__(self, experiment_name="default"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)
        self.run = None

    def start_run(self, run_name=None):
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step=None):
        if not isinstance(metrics, dict):
            if hasattr(metrics, 'box'):
                metrics = metrics.box
            if hasattr(metrics, 'map50') and hasattr(metrics, 'map'):
                metrics = {'mAP50': metrics.map50, 'mAP': metrics.map}
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, file_path):
        mlflow.log_artifact(file_path)

    def end_run(self):
        mlflow.end_run() 
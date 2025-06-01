from configs import get_config
from train import train_and_evaluate
from mlflow_manager import MLflowManager

def main():
    config = get_config()
    mlflow_mgr = MLflowManager(experiment_name="YOLO-AutoML")
    mlflow_mgr.start_run(run_name=config['run_name'])
    mlflow_mgr.log_params(config)
    metrics, best_ckpt = train_and_evaluate(config, return_metrics=True)
    mlflow_mgr.log_metrics(metrics)
    mlflow_mgr.log_artifact(best_ckpt)
    mlflow_mgr.end_run()
    print("실험 완료! 결과:", metrics)
    print("Best checkpoint:", best_ckpt)

if __name__ == "__main__":
    main() 
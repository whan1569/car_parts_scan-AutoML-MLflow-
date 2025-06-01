from train import train_and_evaluate
from cutoff import should_stop_early
from configs import get_config

# 샘플 하이퍼파라미터 후보
HYPERPARAM_CANDIDATES = [
    {'lr': 0.01, 'batch_size': 8},
    {'lr': 0.001, 'batch_size': 16},
    {'lr': 0.005, 'batch_size': 8},
]

def run_automl():
    best_metric = -1
    best_params = None
    for idx, hp in enumerate(HYPERPARAM_CANDIDATES):
        config = get_config()
        config.update(hp)
        config['run_name'] = f"auto_trial_{idx}"
        metrics, ckpt = train_and_evaluate(config)
        val_metric = metrics.get('metrics/mAP_0.5', 0)  # 예시: mAP
        if should_stop_early(val_metric, threshold=0.1):
            print(f"Trial {idx} 조기 종료")
            continue
        if val_metric > best_metric:
            best_metric = val_metric
            best_params = hp
    print("Best params:", best_params)
    return best_params

if __name__ == "__main__":
    run_automl() 
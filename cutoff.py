def should_stop_early(metric, threshold=0.1):
    """
    metric이 threshold 이하이면 True 반환(조기 종료)
    """
    return metric < threshold

# Optuna Pruner 연동용 placeholder
# def optuna_pruner_callback(...):
#     pass 
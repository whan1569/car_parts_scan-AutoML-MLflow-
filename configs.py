def get_config():
    """
    학습 및 평가에 필요한 설정값을 반환합니다.
    """
    config = {
        'model_path': 'yolov8n.pt',  # ultralytics 기본 모델
        'data_path': './carparts_dataset/carparts-seg/data.yaml',  # 데이터셋 yaml 경로
        'epochs': 100,  # Early Stopping이 있으므로 충분히 큰 값으로 설정
        'batch_size': 8,
        'img_size': 640,
        'device': '0',  # GPU 사용
        'workers': 8,  # 데이터 로딩 워커 수
        'optimizer': 'auto',  # 자동 최적화
        'seed': 42,  # 재현성을 위한 시드값
        'output_dir': './runs',
        'run_name': 'exp1',
        
        # Early Stopping 설정
        'early_stopping': {
            'patience': 10,  # 성능 개선이 없을 때 기다리는 에포크 수
            'min_delta': 0.001,  # 최소 개선 기준값
            'mode': 'max',  # 'min' 또는 'max' (손실 최소화 또는 메트릭 최대화)
            'overfitting_threshold': 0.1  # 과적합 판단 기준
        }
    }
    return config 
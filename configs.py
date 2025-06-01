def get_config():
    """
    학습 및 평가에 필요한 설정값을 반환합니다.
    """
    config = {
        'model_path': 'yolov8n.pt',  # ultralytics 기본 모델
        'data_path': './carparts_dataset/carparts-seg/data.yaml',  # 데이터셋 yaml 경로
        'epochs': 2,
        'batch_size': 8,
        'img_size': 640,
        'device': '0',  # GPU 사용
        'workers': 8,  # 데이터 로딩 워커 수
        'optimizer': 'auto',  # 자동 최적화
        'seed': 42,  # 재현성을 위한 시드값
        'output_dir': './runs',
        'run_name': 'exp1',
    }
    return config 
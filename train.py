import os
from ultralytics import YOLO

# 데이터셋 경로, 하이퍼파라미터 등 configs에서 불러오기
from configs import get_config

def train_and_evaluate(config: dict, return_metrics: bool = False) -> tuple:
    """
    YOLO 모델 학습 및 평가를 수행하는 함수
    
    Args:
        config (dict): 학습 설정이 담긴 딕셔너리
        return_metrics (bool): 메트릭을 반환할지 여부
        
    Returns:
        tuple or str: return_metrics가 True면 (metrics, best_ckpt), False면 best_ckpt만 반환
    """
    # YOLO 모델 초기화
    model = YOLO(config['model_path'])
    
    # 데이터셋 로드 (불필요, 삭제)
    # dataset = model.load_dataset(config['data_path'])
    
    # 학습 수행
    results = model.train(
        data=config['data_path'],
        epochs=config['epochs'],
        imgsz=config['img_size'],
        batch=config['batch_size'],
        device=config['device'],
        workers=config['workers'],
        project=config['output_dir'],
        name=config['run_name'],
        exist_ok=True,
        pretrained=True,
        optimizer=config['optimizer'],
        verbose=True,
        seed=config['seed'],
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        cache=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True
    )
    
    # 실제 저장된 디렉토리
    save_dir = results.save_dir if hasattr(results, 'save_dir') else config['output_dir']
    best_ckpt = os.path.join(save_dir, 'weights', 'best.pt')
    
    if return_metrics:
        # 메트릭 추출
        metrics = {
            'mAP50': results.results_dict.get('metrics/mAP50(B)', 0.0),
            'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0.0),
            'precision': results.results_dict.get('metrics/precision(B)', 0.0),
            'recall': results.results_dict.get('metrics/recall(B)', 0.0),
            'box_loss': results.results_dict.get('train/box_loss', 0.0),
            'cls_loss': results.results_dict.get('train/cls_loss', 0.0),
            'dfl_loss': results.results_dict.get('train/dfl_loss', 0.0)
        }
        return metrics, best_ckpt
    
    return best_ckpt

if __name__ == "__main__":
    config = get_config()
    best_ckpt = train_and_evaluate(config)  # 메트릭 없이 실행
    print("Best checkpoint saved at:", best_ckpt) 
from typing import Dict, List, Optional
import numpy as np

class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max',
        overfitting_threshold: float = 0.1
    ):
        """
        Early Stopping 구현 클래스
        
        Args:
            patience (int): 성능 개선이 없을 때 기다리는 에포크 수
            min_delta (float): 최소 개선 기준값
            mode (str): 'min' 또는 'max' (손실 최소화 또는 메트릭 최대화)
            overfitting_threshold (float): 과적합 판단 기준
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.overfitting_threshold = overfitting_threshold
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # 과적합 감지를 위한 변수들
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
    
    def check_epoch(self, metrics: Dict[str, float]) -> bool:
        """
        현재 에포크의 메트릭을 체크하여 조기 종료 여부를 결정
        
        Args:
            metrics (Dict[str, float]): 현재 에포크의 메트릭
            
        Returns:
            bool: True면 학습 중단, False면 계속
        """
        # 메인 메트릭 선택 (mAP50-95)
        current_score = metrics.get('mAP50-95', 0.0)
        
        # 과적합 체크
        train_loss = metrics.get('box_loss', 0.0) + metrics.get('cls_loss', 0.0)
        val_loss = metrics.get('val_box_loss', 0.0) + metrics.get('val_cls_loss', 0.0)
        
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        if len(self.train_losses) > 1:
            # 과적합 체크: 검증 손실이 훈련 손실보다 크게 증가
            if (self.val_losses[-1] - self.train_losses[-1]) > self.overfitting_threshold:
                print("과적합 감지: 학습 중단")
                return True
        
        if self.best_score is None:
            self.best_score = current_score
        elif self.mode == 'min':
            if current_score < self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"성능 개선 없음: {self.patience} 에포크 동안 개선 없음")
                    return True
            else:
                self.best_score = current_score
                self.counter = 0
        else:  # mode == 'max'
            if current_score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"성능 개선 없음: {self.patience} 에포크 동안 개선 없음")
                    return True
            else:
                self.best_score = current_score
                self.counter = 0
        
        return False
    
    def reset(self):
        """상태 초기화"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.train_losses = []
        self.val_losses = []

# Optuna Pruner 연동용 함수
def optuna_pruner_callback(study, trial):
    """
    Optuna Pruner 콜백 함수
    
    Args:
        study: Optuna study 객체
        trial: 현재 trial 객체
    """
    # TODO: Optuna Pruner 구현
    pass 
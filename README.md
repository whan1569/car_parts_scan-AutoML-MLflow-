# Car Parts Detection with AutoML and MLflow

자동차 부품 검출을 위한 YOLO 기반 AutoML 프로젝트입니다. MLflow를 통한 실험 관리와 하이퍼파라미터 최적화를 지원합니다.

## 주요 기능

- YOLOv8 기반 객체 검출 모델 학습
- MLflow를 통한 실험 관리 및 메트릭 추적
- Optuna 기반 하이퍼파라미터 최적화
- 조기 종료(Early Stopping) 기능
- 통합 실험 워크플로우

## 설치 방법

1. 저장소 클론:
```bash
git clone [repository-url]
cd car_parts_scan
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 기본 학습 실행
```bash
python train.py
```

### 2. AutoML 실행
```bash
python main.py
```

## 프로젝트 구조

```
.
├── train.py              # YOLO 모델 학습 및 평가
├── mlflow_manager.py     # MLflow 실험 관리
├── automl.py            # 하이퍼파라미터 최적화
├── cutoff.py            # 조기 종료 기능
├── main.py              # 통합 워크플로우
├── configs.py           # 설정 관리
├── utils.py             # 유틸리티 함수
├── models/              # 학습된 모델 저장 디렉토리
│   └── yolov8n/        # YOLOv8n 모델 관련 파일
├── carparts_dataset/    # 데이터셋
│   └── carparts-seg/   # 세그멘테이션 데이터셋
│       ├── train/
│       ├── valid/
│       ├── test/
│       └── data.yaml
├── mlruns/             # MLflow 실험 기록 (자동 생성)
│   └── [experiment_id]/
│       └── [run_id]/
│           ├── metrics/
│           ├── params/
│           └── artifacts/
└── runs/               # YOLO 학습 결과 (자동 생성)
    └── exp[number]/   # 실험 번호별 결과
        ├── weights/   # 모델 가중치
        ├── results.csv
        └── plots/     # 학습 그래프
```

## 디렉토리 설명

### 자동 생성되는 디렉토리

1. `mlruns/`
   - MLflow가 실험 기록을 저장하는 디렉토리
   - 각 실험의 파라미터, 메트릭, 아티팩트가 저장됨
   - MLflow UI로 확인 가능

2. `runs/`
   - YOLO 학습 과정에서 생성되는 결과물 저장 디렉토리
   - exp1, exp2 등 실험 번호별로 하위 디렉토리 생성
   - weights/: 학습된 모델 가중치
   - results.csv: 학습 메트릭 기록
   - plots/: 학습 과정 그래프

### 수동 관리 디렉토리

1. `models/`
   - 학습된 모델을 저장하는 디렉토리
   - YOLOv8n 등 모델별 하위 디렉토리 구성

2. `carparts_dataset/`
   - 학습에 사용되는 데이터셋
   - train/, valid/, test/ 분할 포함

## 실험 결과

MLflow UI를 통해 실험 결과를 확인할 수 있습니다:
```bash
mlflow ui
```

## 라이선스

MIT License 

# Sparse Graph Random Neural Network (SGRNN) 기반 대화 감정 인식

**저자:** 김윤주  
**소속:** 숙명여자대학교 IT공학과  
**논문:** _Sparse Graph Random Neural Network for Scalable Conversational Emotion Analysis_

## 개요

본 저장소는 **SGRNN + AIM**을 활용한 **대화 감정 인식 (Conversational Emotion Recognition, CER)** 연구 코드입니다.

주요 특징:
- **화자 인지형 희소 그래프(GFPush)**로 긴 거리 의존성 효율적 모델링
- **RoBERTa**, **openSMILE** 기반 멀티모달 특성 추출
- **AIM 모듈**로 시퀀스와 그래프 정보를 동적으로 융합

IEMOCAP과 CREMA-D에서 기존 GNN 기반 모델보다 우수한 성능을 기록하였습니다.

## 폴더 구조

```
.
├── data/ # 업로드 용량의 문제로 github에 업로드 불가, utils/build_dataset.py 에서 데이터셋 다운로드 가능
│   ├── CREMA-D/
│   ├── IEMOCAP/
│   ├── pts/
│   ├── working/
├── utils/
│   ├── dataset_loader.py
│   ├── feature_extractor.py
│   ├── preprocess.py
├── models/
│   ├── modules/
│   │   ├── lstm_encoder.py
│   │   ├── graph_constructor.py
│   │   ├── graph_random_network.py
│   │   ├── aim_module.py
│   ├── elr_gnn.py
├── train.py
├── test.py
└── config.yaml
```

## ⚙️ 요구사항

- Python >= 3.11
- PyTorch >= 2.7
- transformers, opensmile, scikit-learn, seaborn, matplotlib

설치:

```bash
pip install torch transformers opensmile scikit-learn seaborn matplotlib
```

## 실행 방법

[1] `config.yaml` 설정  
[2] 데이터 생성: `build_dataset.ipynb 실행`
[3] 전처리: `python preprocess.py`  
[4] 학습: `python train.py`  
[5] 평가: `python test.py`

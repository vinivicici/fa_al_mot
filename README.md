# 👕 패알몯(패션 가격 알려주는 모델)

> **“그 옷, 얼마에 팔았어?”**  
> **멀티모달 딥러닝 기반 C2C 패션 중고 거래 적정 가격 예측 및 속성 분류 솔루션**

---

## 📖 프로젝트 개요 (Overview)

온라인 C2C(개인 간 거래) 패션 시장은 빠르게 성장하고 있지만,  
여전히 **정보의 비대칭성**과 **가격 책정의 불확실성**이라는 문제가 존재합니다.

판매자는 적절한 판매가를 결정하기 어렵고,  
구매자는 제시된 가격이 합리적인지 판단하기 힘듭니다.

이에 패알몯 프로젝트는 의류 상품의
**이미지(Image)** 와 **텍스트 설명(Text / Description)** 을 결합한  
**멀티모달 AI 모델**을 통해 보다 **투명하고 신뢰 가능한 중고 패션 거래 환경**을 구축하는 것을 목표로 합니다.

---

## 🏆 주요 성과 (Key Achievements)

### 🔹 가격 예측 성능 향상
- **$R^2 = 0.89$** 달성
- 기존 머신러닝 모델($R^2 \approx 0.68$) 대비 **오차 약 18% 감소**
- LLM 기반 전처리, Raw Embedding 활용, Advanced MLP 도입 효과 입증

### 🔹 상품 속성 분류 SOTA 성능
- **CLIP Fusion 모델** 활용
- Category / Gender / Usage 전 태스크에서 최고 성능 기록

### 🔹 데이터 품질 고도화
- **Meta-Llama-3** 기반 LLM 적용
- 비정형 상품 설명에서 Brand, Material 등 핵심 속성 자동 복원

### 🔹 학습 효율 극대화
- GPU 가속 및 Mixed Precision 적용
- 학습 시간 **35분(CPU) → 3분(GPU)**  
- 약 **11배 학습 속도 개선**

---

## 📂 데이터셋 (Dataset)

성격이 다른 두 개의 대규모 오픈 데이터셋을 결합하여  
**가격 예측 및 속성 분류에 최적화된 통합 데이터셋**을 구축했습니다.

### 사용 데이터셋
1. **H&M Personalized Fashion Recommendations**
2. **Fashion Product Images Dataset**

### 데이터 처리 파이프라인
- 가격 스케일 통합 (H&M 가격 복원 및 Fashion 데이터 정규화)
- 카테고리 / 성별 / 용도 매핑 및 통합
- LLM 기반 결측치 보완 및 속성 추출

---

## 🚀 주요 기술 및 방법론 (Methodologies)

### 1. LLM 기반 데이터 전처리
- **Model:** `Meta-Llama-3.1-8B-Instruct`
- **역할:** 상품 설명에서 가격 결정 핵심 속성 정밀 추출
- **Code:**  
  `Code for LLM, Classification, Regression/Use_LLM_for_preprocess.py`

### 2. 멀티모달 클러스터링
- CLIP 기반 이미지 + 텍스트 임베딩 활용
- **Methods:** K-Means, GMM, Agglomerative, **DEC**
- **Code:**  
  `Code for Embedding, Clustering/`

### 3. 상품 속성 분류
- **Tasks:** Category(11), Gender(5), Usage(8)
- **Models:** XGBoost, LightGBM, CNN, **CLIP Fusion**
- **Code:**  
  `Code for LLM, Classification, Regression/Classification_with_CLIP.py`

### 4. 가격 예측 (Core Task)
- **Approach:** Raw Embedding 직접 활용 (PCA 미적용)
- **Model:** **Advanced MLP** (GPU + Mixed Precision)
- **Performance:** $R^2 = 0.8904$
- **Code:**  
  `Code for LLM, Classification, Regression/regression_DL_SOTA.py`

---

## 🛠 설치 및 실행 (Installation & Usage)

### 환경 설정
- Python 3.8+
- NVIDIA GPU 권장

```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch torchvision transformers open_clip_torch lightgbm xgboost tqdm
```

### 데이터 전처리
```bash
python preprocess.py
python "Code for LLM, Classification, Regression/Use_LLM_for_preprocess.py"
```

### 분류 모델 학습
```bash
python "Code for LLM, Classification, Regression/Classification_with_CLIP.py"
```

### 가격 예측 모델 학습
```bash
python "Code for LLM, Classification, Regression/regression_DL_SOTA.py"
```

## 📁 프로젝트 구조
```
fa_al_mot/
├── dataset/
├── utils/
├── Code for Embedding, Clustering/
├── Code for LLM, Classification, Regression/
├── code_for_image_captioning/
├── preprocess.py
└── README.md
```

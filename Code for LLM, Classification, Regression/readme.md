

# 👕 패알몬

> **AI 기반 C2C 패션 중고 거래 적정 가격 예측 및 속성 분류 솔루션**

---

## 📖 개요 (Overview)

이미지(Image)와 텍스트(Text) 정보를 결합한 **멀티모달 딥러닝**을 활용하여
상품의 **카테고리, 성별, 용도**를 자동으로 분류하고,
데이터에 기반한 **객관적인 적정 중고 가격(USD)** 을 예측합니다.

---

## 📂 데이터셋 (Dataset)

데이터의 양과 질을 확보하기 위해 서로 다른 두 개의 대형 오픈 데이터셋을 병합하여 사용했습니다.

1. **H&M Personalized Fashion Recommendations**
   대규모 거래 내역 및 상품 이미지 포함

2. **Fashion Product Images Dataset (Kaggle)**
   인도 패션 시장 데이터, 상세한 속성(Category, Usage 등) 포함

3. **데이터 전처리**

   * 이상치 제거
   * 카테고리 재매핑
   * **LLM 기반 속성 추출**

---

## 🚀 주요 기술 및 방법론 (Key Methodologies)

### 1. LLM 기반 데이터 전처리 (Advanced Preprocessing)

기존의 불완전하고 비정형적인 상품 설명(Description)을
구조화된 데이터로 변환하기 위해 생성형 AI를 도입했습니다.

* **모델:** `Meta-Llama-3.1-8B-Instruct`
* **역할:** 텍스트에서 **Brand, Material, Care, Style** 등
  가격 결정 핵심 인자를 정밀 추출
* **성과:** 결측치 보완 및 데이터 완전성(Completeness) 확보
* **관련 파일:** `Use_LLM_for_preprocess.py`

---

### 2. 멀티모달 클러스터링 (Clustering)

메타데이터가 부족한 환경을 가정하여,
임베딩 정보만으로 상품 군집을 형성하는 실험을 진행했습니다.

* **비교:**
  ML (K-Means, GMM) vs DL (DEC: Deep Embedded Clustering)
* **결과:**
  딥러닝 기반 **DEC**가 비선형적 맥락을 학습하여
  가장 뚜렷한 군집 경계를 형성

---

### 3. 상품 속성 분류 (Classification)

상품의 **Category(11종), Gender(5종), Usage(8종)** 를 예측합니다.

* **Machine Learning**

  * XGBoost, LightGBM, LinearSVC 등 8종 모델 실험
  * (`Classification_ML.py`)
* **Deep Learning**

  * CNN (ResNet50)
  * Multimodal CNN
  * **CLIP Fusion**
* **SOTA 달성**

  * **CLIP Fusion 모델**이 이미지–텍스트 정렬(Alignment) 능력을 바탕으로
    모든 태스크에서 최고 성능 달성
  * Usage Accuracy: **0.91 (CLIP)** vs 0.58 (CNN Image-only)

---

### 4. 가격 예측 (Price Regression) — **Core Task**

고차원 임베딩 데이터를 활용하여 적정 가격(USD)을 산출합니다.

* **Machine Learning**

  * LightGBM: $R^2$ **0.8364**
* **Deep Learning (SOTA)**

  * **Advanced MLP (GPU 가속 + Mixed Precision)**
  * **전략**

    * PCA 없이 Raw Embedding을 그대로 학습하여 정보 손실 최소화
  * **결과**

    * **$R^2$ 0.8904** 달성
      (기존 ML 대비 오차 18% 감소)
  * **효율**

    * 학습 시간: 35분(CPU) → **3분(GPU)**
      약 11배 단축

---

## 📊 실험 결과 요약 (Performance Summary)

### Classification Performance (F1-Score)

| Model           | Type   | Gender   | Category | Usage    | 비고                   |
| --------------- | ------ | -------- | -------- | -------- | -------------------- |
| **CLIP Fusion** | **DL** | **0.86** | **0.83** | **0.50** | **Best Performance** |
| XGBoost         | ML     | 0.85     | 0.79     | 0.51     | High Efficiency      |
| Multimodal CNN  | DL     | 0.68     | 0.68     | 0.39     |                      |
| ResNet50 (Img)  | DL     | 0.53     | 0.53     | 0.37     |                      |

---

### Regression Performance ($R^2$ & Time)

| Experiment           | Model                 | $R^2$      | RMSE       | Time (sec) | 비고                |
| -------------------- | --------------------- | ---------- | ---------- | ---------- | ----------------- |
| **Exp 1 (Meta+Emb)** | **Advanced MLP (DL)** | **0.8904** | **382.63** | **190s**   | **SOTA, Fastest** |
| Exp 1 (Meta+Emb)     | LightGBM (ML)         | 0.8364     | 467.57     | 2,133s     |                   |
| Exp 1 (Meta+Emb)     | Linear Reg (ML)       | 0.7456     | 583.04     | 74s        | Baseline          |

> **Insight**
> 고차원 패션 데이터(Embedding) 처리에는
> **GPU 기반 딥러닝**이 성능과 효율성 모두에서
> 머신러닝(Tree 기반)을 압도함을 확인했습니다.

---

## 🛠 설치 및 실행 방법 (Installation & Usage)

### 1. 환경 설정 (Prerequisites)

본 프로젝트는 **Python 3.8+** 환경에서 동작하며,
DL 모델 학습을 위해 **NVIDIA GPU** 사용을 권장합니다.

```bash
pip install torch torchvision transformers lightgbm scikit-learn pandas numpy tqdm open_clip_torch
```

---

### 2. 데이터 전처리 (LLM 기반)

```bash
python Use_LLM_for_preprocess.py
```

---

### 3. 분류 모델 학습 (Classification)

```bash
python Classification_with_CLIP.py
```

* ML 베이스라인:

```bash
python Classification_ML.py
```

---

### 4. 가격 예측 모델 학습 (Regression)

```bash
python regression_DL_SOTA.py
```

* ML 베이스라인:

```bash
python regression_ML.py
```

---

## 📂 파일 구조 (File Structure)

```
📦 Fashion-Price-Informant
 ┣ 📜 Use_LLM_for_preprocess.py
 ┣ 📜 Classification_with_CLIP.py
 ┣ 📜 Classification_with_CNN.py
 ┣ 📜 Classification_with_CNN_multimodal.py
 ┣ 📜 Classification_ML.py
 ┣ 📜 regression_DL_SOTA.py
 ┗ 📜 regression_ML.py
```


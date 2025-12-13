import pandas as pd
import numpy as np
import time  # 시간 측정을 위한 모듈
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb  # LightGBM 임포트

# ---------------------------------------------------------
# 1. 설정 및 데이터 로드
# ---------------------------------------------------------
CSV_PATH = ""
SAVE_PATH = "regression_results_lgbm_ridge_linear.csv" # 파일명 변경

print(f"Loading data from {CSV_PATH}...")
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print("Error: 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# 타겟 변수(price_usd)에 NaN이 있는 행 제거
initial_rows = len(df)
df = df.dropna(subset=['price_usd'])
print(f"Dropped {initial_rows - len(df)} rows with missing 'price_usd'. Remaining rows: {len(df)}")

if len(df) == 0:
    print("Error: 모든 데이터의 price_usd 값이 비어있습니다.")
    exit()

# ---------------------------------------------------------
# 2. Feature Set 준비
# ---------------------------------------------------------

# (1) Metadata Feature Set
metadata_cols = [c for c in df.columns if 'normalized' in c]
X_metadata = df[metadata_cols]

# (2) Cluster Feature Set
if 'cluster' in df.columns:
    X_cluster = pd.get_dummies(df['cluster'], prefix='cluster')
else:
    X_cluster = pd.DataFrame(index=df.index)

# (3) Embedding Feature Set
def parse_embedding_column(series):
    def convert(x):
        try:
            if isinstance(x, str):
                return np.fromstring(x, sep=',')
            return np.zeros(0)
        except:
            return np.zeros(0)
    
    parsed_list = series.apply(convert)
    valid_sample = next((x for x in parsed_list if x.size > 0), None)
    
    if valid_sample is None:
        return None

    try:
        # 데이터가 균일한 차원을 가지도록 보장
        dim = valid_sample.shape[0]
        fixed_list = [x if x.shape[0] == dim else np.zeros(dim) for x in parsed_list]
        return np.vstack(fixed_list)
    except ValueError:
        return None

emb_cols = ['img_emb', 'wrd_emb', 'emb_256']
emb_matrices = []

print("Parsing embeddings...")
for col in emb_cols:
    if col in df.columns:
        matrix = parse_embedding_column(df[col])
        if matrix is not None:
            emb_matrices.append(matrix)

if emb_matrices:
    X_embeddings_np = np.hstack(emb_matrices)
    # 컬럼명 생성
    X_embeddings = pd.DataFrame(
        X_embeddings_np, 
        columns=[f'emb_{i}' for i in range(X_embeddings_np.shape[1])], 
        index=df.index
    )
else:
    X_embeddings = pd.DataFrame(index=df.index)

# ---------------------------------------------------------
# 3. 실험 데이터셋 구성
# ---------------------------------------------------------
experiments = {}

if not X_metadata.empty and not X_embeddings.empty:
    experiments["Exp 1: Metadata + Embedding"] = pd.concat([X_metadata, X_embeddings], axis=1)

if not X_cluster.empty and not X_embeddings.empty:
    experiments["Exp 2: Cluster + Embedding"] = pd.concat([X_cluster, X_embeddings], axis=1)

target = df['price_usd']

# ---------------------------------------------------------
# 4. 모델 정의 (요청하신 모델들로 변경)
# ---------------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "LGBM": lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1)
}

# ---------------------------------------------------------
# 5. 학습, 평가 및 시간 측정
# ---------------------------------------------------------
results = []

print(f"\n{'Experiment':<35} | {'Model':<20} | {'RMSE':<10} | {'R2 Score':<10} | {'Time(s)':<10}")
print("-" * 100)

for exp_name, X_data in experiments.items():
    # LGBM은 컬럼명에 특수문자가 있으면 에러가 날 수 있으므로 처리 (JSON 포맷 등 제거)
    import re
    X_data = X_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    X_data = X_data.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X_data, target, test_size=0.2, random_state=42)
    
    for model_name, model in models.items():
        try:
            # 시간 측정 시작
            start_time = time.time()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # 시간 측정 종료
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"{exp_name:<35} | {model_name:<20} | {rmse:<10.4f} | {r2:<10.4f} | {elapsed_time:<10.4f}")
            
            results.append({
                "Experiment": exp_name,
                "Model": model_name,
                "RMSE": rmse,
                "R2_Score": r2,
                "Time_sec": elapsed_time
            })
            
        except Exception as e:
            print(f"{exp_name:<35} | {model_name:<20} | Error: {e}")
            results.append({
                "Experiment": exp_name,
                "Model": model_name,
                "RMSE": None,
                "R2_Score": None,
                "Time_sec": None,
                "Error": str(e)
            })

# ---------------------------------------------------------
# 6. 결과 CSV 저장
# ---------------------------------------------------------
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(SAVE_PATH, index=False)
    print(f"\nResults (including time) have been saved to '{SAVE_PATH}'")
else:
    print("\nNo results to save.")
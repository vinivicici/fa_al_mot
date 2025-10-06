import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# data_loader.py에서 데이터 로딩 함수와 파일 경로를 가져옵니다.
from data_loader import load_data, get_regression_data, PARQUET_FILE_PATH

def preprocess_features_regression(X_train, X_test):
    """
    회귀 작업을 위해 훈련/테스트 데이터셋의 피처를 전처리합니다.
    - Vector 컬럼 확장
    - Categorical 컬럼(style)을 One-Hot Encoding
    """
    # 훈련/테스트 세트를 잠시 합쳐서 컬럼 구조를 일치시킴 (Data Leakage 방지)
    combined = pd.concat([X_train, X_test], keys=['train', 'test'])
    
    # 1. 'image_embedding_vector' 확장
    if 'image_embedding_vector' in combined.columns:
        print("  'image_embedding_vector' 컬럼을 확장하여 피처로 변환합니다...")
        embedding_df = pd.DataFrame(
            combined['image_embedding_vector'].tolist(), 
            index=combined.index
        )
        embedding_df.columns = [f'embed_{i}' for i in range(embedding_df.shape[1])]
        combined = combined.drop('image_embedding_vector', axis=1)
        combined = pd.concat([combined, embedding_df], axis=1)

    # 전처리가 완료된 데이터셋을 다시 훈련/테스트 용으로 분리
    X_train_processed = combined.loc['train']
    X_test_processed = combined.loc['test']
    
    return X_train_processed, X_test_processed


def main():
    print("="*50)
    print("Regression Task Start")
    print("="*50)

    # 1. data_loader를 사용해 Parquet 파일에서 데이터를 불러옵니다.
    df = load_data(PARQUET_FILE_PATH)
    if df is None:
        print("데이터 로딩에 실패하여 회귀 분석을 종료합니다.")
        return
        
    # 2. 회귀용 데이터셋을 생성하고 분리합니다.
    X_train_raw, X_test_raw, y_train, y_test = get_regression_data(df)
    if X_train_raw is None:
        print("회귀 데이터 준비에 실패하여 작업을 종료합니다.")
        return
        
    # 3. 훈련 데이터와 테스트 데이터의 피처를 전처리합니다.
    print("\n[INFO] 피처 전처리(Feature Preprocessing)를 시작합니다.")
    X_train, X_test = preprocess_features_regression(X_train_raw, X_test_raw)
    print(f"전처리 후 훈련 데이터 형태: {X_train.shape}")
    print(f"전처리 후 테스트 데이터 형태: {X_test.shape}")


    # 회귀 모델 정의
    models = {
        "Linear Regression (OLS)": LinearRegression(n_jobs=-1),
        "Ridge Regression": Ridge(random_state=42),
        "Lasso Regression": Lasso(random_state=42),
        # SVR은 데이터셋이 크면 매우 느릴 수 있습니다.
        # "Support Vector Machine (SVR)": SVR(),
        "Random Forest Regressor": RandomForestRegressor(random_state=42, n_jobs=-1),
        "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_jobs=-1)
    }

    # 각 모델에 대해 학습 및 평가 진행
    for name, model in models.items():
        print(f"\n--- Training and Evaluating {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 1. Metrics 출력
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"  R² (결정계수): {r2:.4f} (1에 가까울수록 좋음)")
        print(f"  MSE (평균제곱오차): {mse:.4f}")
        print(f"  RMSE (평균제곱근오차): {rmse:.4f} (0에 가까울수록 좋음)")

        # 2. 회귀 결과 시각화 (Actual vs. Predicted)
        plt.figure(figsize=(7, 7))
        plt.scatter(y_test, y_pred, alpha=0.5, label='Predictions')
        # 완벽한 예측을 나타내는 대각선
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label='Perfect Fit')
        plt.title(f'Actual vs. Predicted Values for {name}\n$R^2 = {r2:.4f}$')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
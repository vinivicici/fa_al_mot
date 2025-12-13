import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

# data_loader.py에서 데이터 로딩 함수와 파일 경로를 가져옵니다.
from data_loader import load_data, get_classification_data, PARQUET_FILE_PATH

def preprocess_features(X):
    """
    DataFrame에서 vector 컬럼을 확장하고, 훈련에 부적합한 컬럼을 제거합니다.
    scikit-learn 모델이 처리할 수 있는 형태로 피처를 변환합니다.
    """
    # 원본 DataFrame 복사
    X_processed = X.copy()
    
    # 'image_embedding_vector'가 있으면 각 원소를 별개의 피처(컬럼)로 확장
    if 'image_embedding_vector' in X_processed.columns:
        print("  'image_embedding_vector' 컬럼을 확장하여 피처로 변환합니다...")
        # 벡터 리스트를 새로운 DataFrame으로 변환
        embedding_df = pd.DataFrame(
            X_processed['image_embedding_vector'].tolist(), 
            index=X_processed.index
        )
        # 확장된 컬럼에 이름 부여 (e.g., embed_0, embed_1, ...)
        embedding_df.columns = [f'embed_{i}' for i in range(embedding_df.shape[1])]
        
        # 기존 벡터 컬럼을 제거하고 확장된 피처를 병합
        X_processed = X_processed.drop('image_embedding_vector', axis=1)
        X_processed = pd.concat([X_processed, embedding_df], axis=1)

    # 혹시 모를 다른 object 타입(문자열 등) 컬럼 제거
    # 이 예제에서는 없지만, 일반적인 상황을 대비한 방어 코드
    non_numeric_cols = X_processed.select_dtypes(include=['object', 'category']).columns
    if not non_numeric_cols.empty:
        print(f"  훈련에 부적합한 비-숫자형 컬럼을 제거합니다: {non_numeric_cols.tolist()}")
        X_processed = X_processed.drop(columns=non_numeric_cols)
        
    return X_processed

def main():
    print("🚀\n" + "="*50)
    print("Classification Task Start")
    print("="*50)

    # 1. data_loader를 사용해 Parquet 파일에서 데이터를 불러옵니다.
    df = load_data(PARQUET_FILE_PATH)
    if df is None:
        print("데이터 로딩에 실패하여 분류 작업을 종료합니다.")
        return

    # 2. 분류용 데이터셋을 생성하고 분리합니다.
    X_train_raw, X_test_raw, y_train, y_test = get_classification_data(df)
    if X_train_raw is None:
        print("분류 데이터 준비에 실패하여 작업을 종료합니다.")
        return
        
    # 3. 훈련 데이터와 테스트 데이터의 피처를 전처리합니다. (Vector 확장 등)
    print("\n[INFO] 피처 전처리(Feature Preprocessing)를 시작합니다.")
    X_train = preprocess_features(X_train_raw)
    X_test = preprocess_features(X_test_raw)
    print(f"전처리 후 훈련 데이터 형태: {X_train.shape}")
    print(f"전처리 후 테스트 데이터 형태: {X_test.shape}")


    # 분류 모델 정의
    models = {
        "Decision Tree (CART)": DecisionTreeClassifier(criterion='gini', random_state=42),
        "Decision Tree (Entropy)": DecisionTreeClassifier(criterion='entropy', random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        # SVM은 고차원 데이터에서 매우 느릴 수 있으므로, 필요시 주석 처리
        # "SVM": SVC(kernel='rbf', random_state=42) 
    }

    # 각 모델에 대해 학습 및 평가 진행
    for name, model in models.items():
        print(f"\n--- Training and Evaluating {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 1. Metrics 출력
        # zero_division=0: 클래스가 하나도 예측되지 않았을 경우 경고 대신 0으로 처리
        print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"  Precision (Macro): {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
        print(f"  Recall (Macro): {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
        print(f"  F1-Score (Macro): {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")

        # 2. Confusion Matrix 시각화
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f'Confusion Matrix for {name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.show()

if __name__ == "__main__":
    main()
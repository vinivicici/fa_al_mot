import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ---  이 부분을 실제 Parquet 파일 경로로 수정 ---
PARQUET_FILE_PATH = 'YOUR_FILE_PATH.parquet'
# ----------------------------------------------------

def load_data(path):
    """지정된 경로에서 Parquet 파일을 불러와 DataFrame으로 반환합니다."""
    print(f"'{path}' 파일에서 데이터를 불러옵니다...")
    try:
        df = pd.read_parquet(path)
        print("데이터 불러오기 완료!")
        # 불러온 데이터의 기본 정보 출력
        '''
        print("\n--- 데이터 정보 ---")
        df.info()
        print("\n--- 데이터 샘플 (상위 5개) ---")
        print(df.head())
        print("-" * 25)
        '''
        return df
    except FileNotFoundError:
        print(f"오류: '{path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return None

def get_clustering_data(df):
    """
    클러스터링 작업을 위해 'image_embedding_vector' 컬럼을 준비.
    클러스터링은 비지도 학습이므로 X 데이터만 반환.
    """
    print("\n클러스터링 데이터 준비 중...")
    if 'image_embedding_vector' not in df.columns:
        print("오류: 'image_embedding_vector' 컬럼을 찾을 수 없습니다.")
        return None

    # 벡터 컬럼을 scikit-learn이 처리할 수 있는 numpy 배열 형태로 변환
    X_cluster = np.vstack(df['image_embedding_vector'].values)
    print("클러스터링 데이터 준비 완료!")
    return X_cluster

def get_classification_data(df):
    """
    'style' 컬럼을 예측하기 위한 분류용 데이터셋을 생성하고 분리합니다.
    - y (타겟): 'style'
    - X (피처): 'style'을 제외한 모든 컬럼
    """
    print("\n분류 데이터 준비 중...")
    if 'style' not in df.columns:
        print("오류: 타겟 컬럼인 'style'을 찾을 수 없습니다.")
        return None, None, None, None

    # 타겟(y)과 피처(X)를 정의
    y = df['style']
    # 예측 대상인 price와 style, 그리고 벡터인 embedding 컬럼을 제외
    X = df.drop(columns=['style'])

    print(f"분류 작업에 사용될 피처: {X.columns.tolist()}")

    # 학습용/테스트용 데이터 분리 (stratify=y 옵션으로 y의 클래스 비율을 유지)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print("분류 데이터 준비 완료!")
    return X_train, X_test, y_train, y_test

def get_regression_data(df):
    """
    'price' 컬럼을 예측하기 위한 회귀용 데이터셋을 생성하고 분리합니다.
    - y (타겟): 'price'
    - X (피처): 'price'를 제외한 모든 컬럼
    """
    print("\n회귀 데이터 준비 중...")
    if 'price' not in df.columns:
        print("오류: 타겟 컬럼인 'price'를 찾을 수 없습니다.")
        return None, None, None, None

    # 타겟(y)과 피처(X)를 정의
    y = df['price']
    X = df.drop(columns=['price'])

    print(f"회귀 작업에 사용될 피처: {X.columns.tolist()}")

    # 학습용/테스트용 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print("회귀 데이터 준비 완료!")
    return X_train, X_test, y_train, y_test
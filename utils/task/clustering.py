import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
# data_loader.py에서 데이터 로딩 함수와 파일 경로를 가져옵니다.
from data_loader import load_data, get_clustering_data, PARQUET_FILE_PATH

def main():
    """
    클러스터링 작업을 수행하고, 각 모델의 성능을 평가한 후 결과를 시각화합니다.
    """
    print("="*50)
    print("Clustering Task Start")
    print("="*50)

    # 1. data_loader를 사용해 Parquet 파일에서 데이터를 불러옵니다.
    #    PARQUET_FILE_PATH는 data_loader.py에 정의된 경로를 사용합니다.
    df = load_data(PARQUET_FILE_PATH)
    if df is None:
        print("데이터 로딩에 실패하여 클러스터링을 종료합니다.")
        return

    # 2. 클러스터링에 사용할 데이터를 준비합니다.
    X = get_clustering_data(df)
    if X is None:
        print("클러스터링 데이터 준비에 실패하여 작업을 종료합니다.")
        return

    print(f"\n원본 데이터 차원: {X.shape}")
    print("시각화를 위해 PCA를 사용하여 2차원으로 축소합니다...")
    
    # 3. 고차원 데이터를 2D로 시각화하기 위해 PCA로 차원 축소
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    print(f"축소된 데이터 차원: {X_2d.shape}")


    # 클러스터링 모델 정의
    # Non-hierarchical Models
    non_hierarchical_models = {
        "K-Means": KMeans(n_clusters=4, random_state=42, n_init=10), #n_clusters는 데이터에 맞게 조정 필요
        "K-Means++": KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10), #n_clusters는 데이터에 맞게 조정 필요
        # Affinity Propagation은 데이터셋이 크면 매우 느릴 수 있습니다.
        # "Affinity Propagation": AffinityPropagation(random_state=42) 
    }

    # Hierarchical Models
    hierarchical_models = {
        "Agglomerative (Single)": AgglomerativeClustering(n_clusters=4, linkage='single'),
        "Agglomerative (Complete)": AgglomerativeClustering(n_clusters=4, linkage='complete'),
        "Agglomerative (Average)": AgglomerativeClustering(n_clusters=4, linkage='average'),
        "Agglomerative (Ward)": AgglomerativeClustering(n_clusters=4, linkage='ward')
    }

    all_models = {**non_hierarchical_models, **hierarchical_models}

    # 각 모델에 대해 클러스터링 실행 및 평가
    for name, model in all_models.items():
        print(f"\n--- Running {name} ---")
        
        # 원본 고차원 데이터(X)로 클러스터링을 학습합니다.
        cluster_labels = model.fit_predict(X)

        # 1. Metric 출력 (Silhouette Score)
        # 1개 이하의 클러스터가 생성되면 실루엣 점수 계산 불가
        if len(set(cluster_labels)) > 1:
            # 실루엣 점수는 원본 고차원 데이터로 계산해야 의미가 있습니다.
            score = silhouette_score(X, cluster_labels)
            print(f"  Silhouette Score: {score:.4f} (1에 가까울수록 좋음)")
        else:
            print("  Cannot calculate Silhouette Score (only 1 or 0 clusters found).")

        # 2. 클러스터링 결과 시각화 (2D로 축소된 데이터 사용)
        plt.figure(figsize=(8, 6))
        # 시각화는 2D로 축소된 데이터(X_2d)를 사용합니다.
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap='viridis', s=20, alpha=0.7)
        plt.title(f'Clustering Result for {name}\n(Visualized with PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(handles=scatter.legend_elements()[0], labels=set(cluster_labels), title="Clusters")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

if __name__ == "__main__":
    main()
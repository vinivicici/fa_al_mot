import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, mixed_precision

# ---------------------------------------------------------
# 0. A5000 GPU 설정 및 Mixed Precision 적용
# ---------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Available: {len(gpus)} (Device: {gpus[0].name})")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ GPU를 찾을 수 없습니다. CPU로 실행됩니다.")

try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"Mixed Precision Policy: {policy.name}")
except Exception as e:
    print(f"Mixed Precision 설정 실패: {e}")

# ---------------------------------------------------------
# 1. 설정 및 데이터 로드
# ---------------------------------------------------------
CSV_PATH = ""
SAVE_CSV_PATH = "a5000_regression_results.csv"
SAVE_IMG_DIR = "vis_results_regression"  # 이미지를 저장할 폴더

if not os.path.exists(SAVE_IMG_DIR):
    os.makedirs(SAVE_IMG_DIR)

BATCH_SIZE = 256 
EPOCHS = 300
LEARNING_RATE = 0.001

print(f"\nLoading data from {CSV_PATH}...")
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print("Error: 파일을 찾을 수 없습니다.")
    exit()

df = df.dropna(subset=['price_usd'])
print(f"Remaining rows: {len(df)}")

if len(df) == 0:
    exit()

# ---------------------------------------------------------
# 2. Feature Set 준비
# ---------------------------------------------------------
metadata_cols = [c for c in df.columns if 'normalized' in c]
X_metadata = df[metadata_cols]

if 'cluster' in df.columns:
    X_cluster = pd.get_dummies(df['cluster'], prefix='cluster')
else:
    X_cluster = pd.DataFrame(index=df.index)

def parse_embedding_column_optimized(series):
    try:
        return np.vstack(series.str.split(',').apply(lambda x: np.array(x, dtype=np.float32)))
    except Exception as e:
        print(f"Vectorized parsing failed: {e}")
        def convert(x):
            try:
                if isinstance(x, str): return np.fromstring(x, sep=',')
                return np.zeros(0)
            except: return np.zeros(0)
        parsed_list = series.apply(convert)
        valid_sample = next((x for x in parsed_list if x.size > 0), None)
        if valid_sample is None: return None
        dim = valid_sample.shape[0]
        fixed_list = [x if x.shape[0] == dim else np.zeros(dim) for x in parsed_list]
        return np.vstack(fixed_list)

emb_cols = ['img_emb', 'wrd_emb', 'emb_256']
emb_matrices = []

print("Parsing embeddings...")
for col in emb_cols:
    if col in df.columns:
        matrix = parse_embedding_column_optimized(df[col])
        if matrix is not None:
            emb_matrices.append(matrix)

if emb_matrices:
    X_embeddings_np = np.hstack(emb_matrices)
    X_embeddings = pd.DataFrame(
        X_embeddings_np, 
        columns=[f'emb_{i}' for i in range(X_embeddings_np.shape[1])], 
        index=df.index
    ).astype('float32')
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

target = df['price_usd'].astype('float32')

# ---------------------------------------------------------
# 4. 모델 정의
# ---------------------------------------------------------
def build_advanced_model(input_dim):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    
    model.add(layers.Dense(512, kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('swish'))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(256, kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('swish'))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(128, kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('swish'))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(64, kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('swish'))
    model.add(layers.Dropout(0.1))
    
    model.add(layers.Dense(1, dtype='float32'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    return model

# ---------------------------------------------------------
# 5. 학습 및 평가 루프
# ---------------------------------------------------------
results = []
vis_data = {} # 시각화를 위한 데이터 저장소

print(f"\n{'Experiment':<35} | {'Model':<20} | {'RMSE':<10} | {'R2 Score':<10} | {'Time(s)':<10}")
print("-" * 100)

for exp_name, X_data in experiments.items():
    X_data = X_data.fillna(0).astype('float32')
    X_train, X_test, y_train, y_test = train_test_split(X_data, target, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_name = "Advanced MLP (A5000)"
    
    try:
        start_time = time.time()
        model = build_advanced_model(X_train_scaled.shape[1])
        
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0)
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        y_pred = model.predict(X_test_scaled, batch_size=BATCH_SIZE, verbose=0).flatten()
        
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
            "Time_sec": elapsed_time,
            "Batch_Size": BATCH_SIZE
        })
        
        # 시각화를 위해 데이터 저장
        vis_data[exp_name] = {
            'history': history.history,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
    except Exception as e:
        print(f"{exp_name:<35} | {model_name:<20} | Error: {e}")

# 결과 CSV 저장
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(SAVE_CSV_PATH, index=False)
    print(f"\nResults saved to '{SAVE_CSV_PATH}'")
else:
    print("\nNo results to save.")

# ---------------------------------------------------------
# 6. 시각화 (Visualization)
# ---------------------------------------------------------
print("\nGenerating visualizations...")

sns.set(style="whitegrid")

# (1) Learning Curve (Loss)
plt.figure(figsize=(12, 6))
for exp_name, data in vis_data.items():
    hist = data['history']
    plt.plot(hist['loss'], label=f"{exp_name} (Train)")
    plt.plot(hist['val_loss'], linestyle='--', label=f"{exp_name} (Val)")

plt.title("Training vs Validation Loss (MSE)")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.savefig(os.path.join(SAVE_IMG_DIR, "learning_curve.png"))
plt.close()

# (2) Actual vs Predicted Scatter Plot
# 데이터가 너무 많으면 보기 힘들 수 있으므로 샘플링하거나 투명도 조절
for exp_name, data in vis_data.items():
    y_test_vals = data['y_test']
    y_pred_vals = data['y_pred']
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_vals, y_pred_vals, alpha=0.3, s=10, color='blue')
    
    # 이상적인 예측선 (y=x)
    min_val = min(y_test_vals.min(), y_pred_vals.min())
    max_val = max(y_test_vals.max(), y_pred_vals.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title(f"Actual vs Predicted: {exp_name}")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    
    safe_name = exp_name.replace(" ", "_").replace(":", "").replace("+", "plus")
    plt.savefig(os.path.join(SAVE_IMG_DIR, f"scatter_{safe_name}.png"))
    plt.close()

# (3) Residual Plot (잔차 분석)
for exp_name, data in vis_data.items():
    y_test_vals = data['y_test']
    y_pred_vals = data['y_pred']
    residuals = y_test_vals - y_pred_vals
    
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=50)
    plt.title(f"Residual Distribution: {exp_name}")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.axvline(0, color='r', linestyle='--')
    
    safe_name = exp_name.replace(" ", "_").replace(":", "").replace("+", "plus")
    plt.savefig(os.path.join(SAVE_IMG_DIR, f"residual_{safe_name}.png"))
    plt.close()

# (4) RMSE & R2 Comparison Bar Chart
if results:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.barplot(x='Experiment', y='RMSE', data=results_df, ax=axes[0], palette='viridis')
    axes[0].set_title("RMSE Comparison (Lower is Better)")
    axes[0].tick_params(axis='x', rotation=15)
    
    sns.barplot(x='Experiment', y='R2_Score', data=results_df, ax=axes[1], palette='magma')
    axes[1].set_title("R2 Score Comparison (Higher is Better)")
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_IMG_DIR, "metrics_comparison.png"))
    plt.close()

print(f"Visualizations saved in '{SAVE_IMG_DIR}' folder.")
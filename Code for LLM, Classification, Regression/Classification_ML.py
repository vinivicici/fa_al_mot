import pandas as pd
import numpy as np
import time
import re
import os
import glob
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder  # <-- 1. THIS IS THE FIX

# --- 1. Configuration ---

# [수정 필요] 데이터셋 경로
CSV_PATH = ""
HNM_IMG_DIR = ""
FASHION_IMG_DIR = ""
RESULTS_CSV_PATH = ""


# --- 2. Helper Functions ---

# [NEW] Copied from cnn.py logic
def build_image_maps():
    """ 
    Scans HNM and FASHION image directories to create a mapping
    from product_id to the actual file path.
    """
    print("Building image path maps...")
    hnm_map = {}
    valid_extensions = {'.jpg', '.jpeg', '.png'}

    for f_path in tqdm(glob.glob(os.path.join(HNM_IMG_DIR, '*')), desc="Scanning HNM"):
        try:
            f_name = os.path.basename(f_path)
            ext = os.path.splitext(f_name)[1].lower()
            if ext in valid_extensions and len(f_name) > 7:
                pid = f_name[1:7]
                hnm_map[pid] = f_path
        except Exception as e:
            print(f"Skipping HNM file {f_path}: {e}")

    fashion_map = {}
    for f_path in tqdm(glob.glob(os.path.join(FASHION_IMG_DIR, '*')), desc="Scanning FASHION"):
        try:
            f_name_no_ext, ext = os.path.splitext(os.path.basename(f_path))
            if ext.lower() in valid_extensions:
                pid = f_name_no_ext
                fashion_map[pid] = f_path
        except Exception as e:
            print(f"Skipping FASHION file {f_path}: {e}")

    print(f"Found {len(hnm_map)} HNM images and {len(fashion_map)} FASHION images.")
    return hnm_map, fashion_map

# [NEW] Helper for image path mapping (from cnn.py)
def get_image_path(row, hnm_map, fashion_map):
    pid = str(row['product_id'])
    source = row.get('dataset_source', '')
    if source == 'HNM':
        return hnm_map.get(pid)
    elif source == 'FASHION':
        return fashion_map.get(pid)
    return None

def parse_emb(x):
    """Helper function to parse string embedding into numpy array."""
    if not isinstance(x, str):
        return None
    try:
        x_clean = re.sub(r'[\n\[\]]', '', x).strip()
        x_clean = re.sub(r'\s+', ',', x_clean)
        x_clean = x_clean.replace(',,', ',')
        return np.fromstring(x_clean, sep=',')
    except Exception as e:
        # Suppress warnings to avoid clutter
        # print(f"Warning: Could not parse embedding string. Error: {e}")
        return None

def get_target_from_onehot(df, prefix):
    """
    Converts a group of one-hot encoded columns (e.g., 'normalized_gender_...')
    into a single multi-class target vector (e.g., [0, 1, 2, 0, ...]).
    """
    target_cols = [col for col in df.columns if col.startswith(prefix)]

    if not target_cols:
        print(f"Error: No columns found with prefix '{prefix}'. Skipping this task.")
        return None, None

    print(f"Found {len(target_cols)} classes for task '{prefix}': {target_cols}")
    one_hot_data = df[target_cols].values
    class_names = [col.replace(prefix, '') for col in target_cols]
    target_labels = np.argmax(one_hot_data, axis=1)

    return target_labels, class_names

# --- 3. Data Loading and Preprocessing [MODIFIED] ---

print(f"Loading data from: {CSV_PATH}")
try:
    df = pd.read_csv(CSV_PATH)
    print("File loaded successfully (comma separated).")
except Exception as e:
    print(f"Failed to load CSV with comma, trying tab... Error: {e}")
    try:
        df = pd.read_csv(CSV_PATH, sep='\t')
        df.columns = df.columns.str.strip()
        print("File loaded successfully with tab separator.")
    except Exception as e2:
        print(f"Fatal Error: Failed to read CSV. {e2}")
        exit()

# [NEW] 1. Filter by existing image path (to match cnn.py)
print("Building image maps to filter data...")
hnm_map, fashion_map = build_image_maps()

# Ensure product_id is string for mapping
if 'product_id' not in df.columns:
     print(f"Fatal Error: 'product_id' column not found.")
     exit()
df['product_id'] = df['product_id'].astype(str)

df['image_path'] = df.apply(lambda row: get_image_path(row, hnm_map, fashion_map), axis=1)

initial_rows = len(df)
df = df.dropna(subset=['image_path'])
# This is the exact log message you wanted to see
print(f"Filtered DataFrame: {len(df)} rows remaining (removed {initial_rows - len(df)} rows with no matching image).")


# [EXISTING] 2. Filter by parsable embeddings
print("Parsing embeddings...")
for col in ['img_emb', 'wrd_emb']:
    if col not in df.columns:
        print(f"Fatal Error: Required column '{col}' not in CSV.")
        exit()
    df[col] = df[col].apply(parse_emb)

initial_rows = len(df)
df = df.dropna(subset=['img_emb', 'wrd_emb'])
# This log now follows the style of your request
print(f"Filtered DataFrame: {len(df)} rows remaining (removed {initial_rows - len(df)} rows with unparsable embeddings).")


# --- 4. Feature (X) Generation ---

print("Creating feature matrix X...")
X_img = np.vstack(df['img_emb']) # Image embedding
X_wrd = np.vstack(df['wrd_emb']) # Word embedding
X = np.hstack([X_img, X_wrd])    # Concatenate features

print(f"Final feature (X) shape: {X.shape}")

# --- 5. SOTA Model Definitions ---

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "LinearSVC": LinearSVC(random_state=42, max_iter=2000, dual=True),
    "XGBoost": XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss'),
    "LightGBM": LGBMClassifier(random_state=42, n_jobs=-1)
}

# --- 6. Main Experiment Loop ---

tasks = ['normalized_gender_', 'normalized_category_', 'normalized_usage_']
all_results = []

for task_prefix in tasks:
    task_name = task_prefix.replace('normalized_', '').replace('_', '')
    print(f"\n{'='*50}")
    print(f"STARTING TASK: {task_name}")
    print(f"{'='*50}")

    # 1. Create Target (y) for this task
    y, class_names = get_target_from_onehot(df, task_prefix)

    if y is None:
        continue

    print(f"Target (y) shape for task '{task_name}': {y.shape}")
    print(f"Original unique y labels: {np.unique(y)}") # Debugging line

    # --- 2. THIS IS THE FIX ---
    # Remap labels to be contiguous [0, 1, ..., k-1] for XGBoost/LGBM
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    print(f"Encoded unique y labels: {np.unique(y)}") # Debugging line

    # 3. Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError:
        print(f"Warning: Cannot stratify task '{task_name}' (likely too few samples per class). Splitting without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4. Inner loop: test all models on this task
    for model_name, model in models.items():
        print(f"\n--- Testing Model: {model_name} on Task: {task_name} ---")

        start_time = time.time()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        end_time = time.time()
        duration = end_time - start_time

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        print(f"Finished in {duration:.2f}s | F1 (Macro): {f1:.4f} | Accuracy: {acc:.4f}")

        all_results.append({
            "Task": task_name,
            "Model": model_name,
            "Accuracy": acc,
            "Precision (Macro)": precision,
            "Recall (Macro)": recall,
            "F1 (Macro)": f1,
            "Run Time (s)": duration
        })

# --- 7. Final Results ---

print(f"\n{'='*50}")
print("ALL EXPERIMENTS COMPLETE")
print(f"{'='*50}")

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(by=["Task", "F1 (Macro)"], ascending=[True, False])

print(results_df.to_string())

results_df.to_csv(RESULTS_CSV_PATH, index=False)
print(f"\nResults saved to {RESULTS_CSV_PATH}")
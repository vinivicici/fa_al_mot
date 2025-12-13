import pandas as pd
import numpy as np
import os
import glob
import time
from tqdm import tqdm
from PIL import Image
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- 1. 설정: 파일 경로 및 하이퍼파라미터 ---

# [수정 필요] 데이터 경로
CSV_PATH = ""
HNM_IMG_DIR = ""
FASHION_IMG_DIR = ""
RESULTS_CSV_PATH = ""


# 모델 하이퍼파라미터
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMG_SIZE = 224 # ResNet 입력 크기

# 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 2. 데이터 전처리 및 준비 ---

def build_image_maps():
    """ 
    HNM과 FASHION 이미지 디렉토리를 스캔하여
    product_id와 실제 파일 경로를 매핑하는 딕셔너리를 생성합니다.
    """
    print("Building image path maps...")
    hnm_map = {}
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    print(f"Scanning HNM directory: {HNM_IMG_DIR}")
    for f_path in tqdm(glob.glob(os.path.join(HNM_IMG_DIR, '*'))):
        try:
            f_name = os.path.basename(f_path)
            ext = os.path.splitext(f_name)[1].lower()
            if ext in valid_extensions and len(f_name) > 7:
                pid = f_name[1:7]
                hnm_map[pid] = f_path
        except Exception as e:
            print(f"Skipping HNM file {f_path}: {e}")

    fashion_map = {}
    print(f"Scanning FASHION directory: {FASHION_IMG_DIR}")
    for f_path in tqdm(glob.glob(os.path.join(FASHION_IMG_DIR, '*'))):
        try:
            f_name_no_ext, ext = os.path.splitext(os.path.basename(f_path))
            if ext.lower() in valid_extensions:
                pid = f_name_no_ext
                fashion_map[pid] = f_path
        except Exception as e:
            print(f"Skipping FASHION file {f_path}: {e}")
            
    print(f"Found {len(hnm_map)} HNM images and {len(fashion_map)} FASHION images.")
    return hnm_map, fashion_map

def parse_emb(x):
    """Helper function to parse string embedding into numpy array."""
    if not isinstance(x, str):
        return None
    try:
        # Clean string: remove brackets, newlines, and ensure single spaces
        x_clean = re.sub(r'[\n\[\]]', '', x).strip()
        # Replace multiple spaces with a single comma
        x_clean = re.sub(r'\s+', ',', x_clean)
        # Handle case where it's already comma-separated
        x_clean = x_clean.replace(',,', ',') 
        return np.fromstring(x_clean, sep=',')
    except Exception as e:
        print(f"Warning: Could not parse embedding string. Error: {e}")
        return None

def load_and_preprocess_data(csv_path, hnm_map, fashion_map):
    """ 
    CSV를 로드하고, 이미지 맵과 매핑하며, 텍스트 임베딩을 파싱합니다.
    """
    print("Loading and preprocessing CSV...")
    
    try:
        df = pd.read_csv(csv_path, sep='\t')
        df.columns = df.columns.str.strip()
        if 'product_id' not in df.columns:
            raise ValueError("Tab separator failed, trying comma.")
    except Exception as e:
        print(f"Info: Failed to read with tab (Error: {e})... Trying with comma.")
        try:
            df = pd.read_csv(csv_path) 
            df.columns = df.columns.str.strip()
        except Exception as read_e:
            print(f"Fatal Error: Failed to read CSV with both tab and comma. Error: {read_e}")
            return pd.DataFrame(), [], [], []

    if 'product_id' not in df.columns:
        print(f"Fatal Error: 'product_id' column not found.")
        print(f"Available columns: {list(df.columns)}")
        raise KeyError("Could not find 'product_id' in CSV columns.")
        
    if 'wrd_emb' not in df.columns:
        print(f"Fatal Error: 'wrd_emb' column not found. Cannot run multi-modal experiment.")
        raise KeyError("Could not find 'wrd_emb' in CSV columns.")

    df['product_id'] = df['product_id'].astype(str)

    # --- 1. Image Path Mapping ---
    def get_image_path(row):
        pid = row['product_id']
        source = row.get('dataset_source', '') 
        if source == 'HNM':
            return hnm_map.get(pid)
        elif source == 'FASHION':
            return fashion_map.get(pid)
        return None

    df['image_path'] = df.apply(get_image_path, axis=1)
    
    initial_rows = len(df)
    df = df.dropna(subset=['image_path'])
    print(f"Filtered DataFrame: {len(df)} rows remaining (removed {initial_rows - len(df)} rows with no matching image).")

    # --- 2. [MODIFIED] Word Embedding Parsing ---
    print("Parsing word embeddings...")
    df['wrd_emb_parsed'] = df['wrd_emb'].apply(parse_emb)
    
    initial_rows = len(df)
    df = df.dropna(subset=['wrd_emb_parsed'])
    print(f"Filtered DataFrame: {len(df)} rows remaining (removed {initial_rows - len(df)} rows with unparsable 'wrd_emb').")

    # --- 3. Target Column Identification ---
    all_cols = df.columns
    gender_cols = [col for col in all_cols if col.startswith('normalized_gender_')]
    category_cols = [col for col in all_cols if col.startswith('normalized_category_')]
    usage_cols = [col for col in all_cols if col.startswith('normalized_usage_')]

    print(f"Identified {len(gender_cols)} gender targets.")
    print(f"Identified {len(category_cols)} category targets.")
    print(f"Identified {len(usage_cols)} usage targets.")
    
    return df, gender_cols, category_cols, usage_cols

# --- 3. PyTorch Dataset 정의 ---

class FashionImageDataset(Dataset):
    """
    [MODIFIED]
    이미지 경로, 워드 임베딩, 3가지 태스크의 레이블을 반환하는
    커스텀 PyTorch Dataset 클래스입니다.
    """
    def __init__(self, dataframe, target_cols_dict, transform=None):
        self.df = dataframe
        self.image_paths = dataframe['image_path'].values
        
        # 3개 그룹의 레이블을 Numpy 배열로 미리 추출
        self.gender_labels = dataframe[target_cols_dict['gender']].values.astype(np.float32)
        self.category_labels = dataframe[target_cols_dict['category']].values.astype(np.float32)
        self.usage_labels = dataframe[target_cols_dict['usage']].values.astype(np.float32)
        
        # [NEW] 워드 임베딩을 Numpy 배열로 미리 추출
        self.wrd_embeddings = np.vstack(dataframe['wrd_emb_parsed'].values).astype(np.float32)
        self.text_emb_dim = self.wrd_embeddings.shape[1] # Get embedding dimension
        print(f"Dataset initialized with text embedding dimension: {self.text_emb_dim}")
        
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Error loading image {img_path}, returning black image. Error: {e}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')
            
        if self.transform:
            image = self.transform(image)
        
        # 2. [NEW] Get Word Embedding
        wrd_emb = torch.tensor(self.wrd_embeddings[idx], dtype=torch.float32)
            
        # 3. Get Labels
        label_gender = torch.tensor(self.gender_labels[idx], dtype=torch.float32)
        label_category = torch.tensor(self.category_labels[idx], dtype=torch.float32)
        label_usage = torch.tensor(self.usage_labels[idx], dtype=torch.float32)
        
        # [MODIFIED] 이미지, 텍스트 임베딩, (레이블 튜플) 반환
        return image, wrd_emb, (label_gender, label_category, label_usage)

# 이미지 변환 (Train: Augmentation 적용, Test: Augmentation 미적용)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 4. [MODIFIED] 모델 정의 (Multi-Modal ResNet50) ---

class MultiModalResNet(nn.Module):
    """
    [MODIFIED]
    ResNet50 (Image)과 Text Embedding을 입력으로 받아 Concatenate한 후,
    3개의 독립적인 분류 헤드를 가진 멀티모달 모델.
    """
    def __init__(self, num_classes_dict, text_emb_dim):
        super(MultiModalResNet, self).__init__()
        
        # 1. ResNet50(ImageNet 가중치) 로드 및 파라미터 동결
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in base_model.parameters():
            param.requires_grad = False
            
        # 2. 특징 추출기 (마지막 FC 레이어 제외)
        num_ftrs_img = base_model.fc.in_features # (e.g., 2048)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        # 3. [MODIFIED] 결합된 피처 차원
        combined_features_dim = num_ftrs_img + text_emb_dim
        print(f"Model Initializing: Image features ({num_ftrs_img}) + Text features ({text_emb_dim}) = Combined ({combined_features_dim})")
        
        # 4. 3개의 분리된 출력 헤드 (입력 차원 수정됨)
        self.fc_gender = nn.Linear(combined_features_dim, num_classes_dict['gender'])
        self.fc_category = nn.Linear(combined_features_dim, num_classes_dict['category'])
        self.fc_usage = nn.Linear(combined_features_dim, num_classes_dict['usage'])

    def forward(self, x_img, x_text):
        # 1. 이미지 특징 추출
        img_features = self.feature_extractor(x_img)
        img_features = torch.flatten(img_features, 1) # [N, 2048, 1, 1] -> [N, 2048]
        
        # 2. [NEW] 이미지 + 텍스트 특징 결합
        # x_text는 이미 [N, text_emb_dim] 형태의 텐서임
        combined_features = torch.cat((img_features, x_text), dim=1) # [N, 2048 + text_emb_dim]
        
        # 3. 3개 헤드로 각각 Forward
        out_gender = self.fc_gender(combined_features)
        out_category = self.fc_category(combined_features)
        out_usage = self.fc_usage(combined_features)
        
        return out_gender, out_category, out_usage

# --- 5. 학습 및 평가 함수 ---

def train_epoch(model, dataloader, criterion, optimizer):
    """ [MODIFIED] 1 에포크 동안 멀티모달 모델을 학습합니다. """
    model.train()
    running_loss = 0.0
    
    # [MODIFIED] Dataloader에서 (img, text, labels) 3개 수신
    for img_inputs, text_inputs, labels_tuple in tqdm(dataloader, desc="Training"):
        img_inputs = img_inputs.to(DEVICE)
        text_inputs = text_inputs.to(DEVICE) # [NEW] 텍스트 입력도 DEVICE로
        
        labels_gender, labels_category, labels_usage = (
            labels_tuple[0].to(DEVICE),
            labels_tuple[1].to(DEVICE),
            labels_tuple[2].to(DEVICE)
        )
        
        optimizer.zero_grad()
        
        # [MODIFIED] 모델에 2개의 입력 전달
        outputs_gender, outputs_category, outputs_usage = model(img_inputs, text_inputs)
        
        loss_gender = criterion(outputs_gender, labels_gender)
        loss_category = criterion(outputs_category, labels_category)
        loss_usage = criterion(outputs_usage, labels_usage)
        
        loss = loss_gender + loss_category + loss_usage
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * img_inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate_model(model, dataloader):
    """ [MODIFIED] 테스트 데이터셋으로 멀티모달 모델을 평가합니다. """
    model.eval()
    
    all_labels_gender, all_preds_gender = [], []
    all_labels_category, all_preds_category = [], []
    all_labels_usage, all_preds_usage = [], []
    
    with torch.no_grad():
        # [MODIFIED] Dataloader에서 (img, text, labels) 3개 수신
        for img_inputs, text_inputs, labels_tuple in tqdm(dataloader, desc="Evaluating"):
            img_inputs = img_inputs.to(DEVICE)
            text_inputs = text_inputs.to(DEVICE) # [NEW] 텍스트 입력도 DEVICE로
            labels_gender, labels_category, labels_usage = labels_tuple
            
            # [MODIFIED] 모델에 2개의 입력 전달
            outputs_gender, outputs_category, outputs_usage = model(img_inputs, text_inputs)
            
            # 1. Gender
            preds_gender = (torch.sigmoid(outputs_gender) > 0.5).float()
            all_labels_gender.append(labels_gender.cpu().numpy())
            all_preds_gender.append(preds_gender.cpu().numpy())
            
            # 2. Category
            preds_category = (torch.sigmoid(outputs_category) > 0.5).float()
            all_labels_category.append(labels_category.cpu().numpy())
            all_preds_category.append(preds_category.cpu().numpy())
            
            # 3. Usage
            preds_usage = (torch.sigmoid(outputs_usage) > 0.5).float()
            all_labels_usage.append(labels_usage.cpu().numpy())
            all_preds_usage.append(preds_usage.cpu().numpy())
            
    all_labels_gender = np.concatenate(all_labels_gender, axis=0)
    all_preds_gender = np.concatenate(all_preds_gender, axis=0)
    all_labels_category = np.concatenate(all_labels_category, axis=0)
    all_preds_category = np.concatenate(all_preds_category, axis=0)
    all_labels_usage = np.concatenate(all_labels_usage, axis=0)
    all_preds_usage = np.concatenate(all_preds_usage, axis=0)
    
    def get_metrics(labels, preds):
        accuracy = accuracy_score(labels, preds)
        precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        return accuracy, precision_macro, recall_macro, f1_macro

    metrics_gender = get_metrics(all_labels_gender, all_preds_gender)
    metrics_category = get_metrics(all_labels_category, all_preds_category)
    metrics_usage = get_metrics(all_labels_usage, all_preds_usage)
    
    results = {
        "Acc (Gender)": metrics_gender[0], "F1 (Gender)": metrics_gender[3],
        "Acc (Category)": metrics_category[0], "F1 (Category)": metrics_category[3],
        "Acc (Usage)": metrics_usage[0], "F1 (Usage)": metrics_usage[3],
        "Precision (Gender)": metrics_gender[1], "Recall (Gender)": metrics_gender[2],
        "Precision (Category)": metrics_category[1], "Recall (Category)": metrics_category[2],
        "Precision (Usage)": metrics_usage[1], "Recall (Usage)": metrics_usage[2],
    }
    
    return results

def run_experiment(exp_name, df_train, df_test, target_cols_dict, num_classes_dict):
    """하나의 실험 시나리오(Train/Test 셋)를 실행하고 결과를 반환합니다."""
    
    print(f"\n--- Starting Experiment: {exp_name} ---")
    start_time = time.time()
    
    # 1. 데이터셋 및 데이터로더
    train_dataset = FashionImageDataset(df_train, target_cols_dict, transform=data_transforms['train'])
    test_dataset = FashionImageDataset(df_test, target_cols_dict, transform=data_transforms['test'])
    
    # [NEW] 데이터셋에서 텍스트 임베딩 차원 가져오기
    text_emb_dim = train_dataset.text_emb_dim 
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. [MODIFIED] 모델, 손실함수, 옵티마이저
    model = MultiModalResNet(num_classes_dict, text_emb_dim).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # 3. 모델 학습
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1} Train Loss (Combined): {train_loss:.4f}")
        
    # 4. 모델 평가
    metrics_dict = evaluate_model(model, test_loader)
    
    end_time = time.time()
    running_time = end_time - start_time
    
    print(f"--- Experiment Finished: {exp_name} ---")
    print(f"Gender Metrics:     Acc={metrics_dict['Acc (Gender)']:.4f}, F1={metrics_dict['F1 (Gender)']:.4f}")
    print(f"Category Metrics:   Acc={metrics_dict['Acc (Category)']:.4f}, F1={metrics_dict['F1 (Category)']:.4f}")
    print(f"Usage Metrics:      Acc={metrics_dict['Acc (Usage)']:.4f}, F1={metrics_dict['F1 (Usage)']:.4f}")
    print(f"Total Running Time: {running_time:.2f} seconds")
    
    final_results = {
        "Experiment": exp_name,
        **metrics_dict,
        "Running Time (s)": running_time
    }
    
    return final_results

# --- 6. 메인 실행 로직 ---

def main():
    # 1. 데이터 준비 (이미지 맵, CSV 로드, 타겟 컬럼 분리, wrd_emb 파싱)
    hnm_map, fashion_map = build_image_maps()
    df, gender_cols, category_cols, usage_cols = load_and_preprocess_data(CSV_PATH, hnm_map, fashion_map)
    
    target_cols_dict = {
        'gender': gender_cols,
        'category': category_cols,
        'usage': usage_cols
    }
    num_classes_dict = {
        'gender': len(gender_cols),
        'category': len(category_cols),
        'usage': len(usage_cols)
    }
    
    if len(df) == 0:
        print("Fatal Error: No data left after filtering. Exiting.")
        return
    if num_classes_dict['gender'] == 0 or num_classes_dict['category'] == 0 or num_classes_dict['usage'] == 0:
        print("Fatal Error: One of the target groups (gender, category, usage) has 0 columns. Check CSV column names. Exiting.")
        return

    # 2. 실험별 데이터 분리
    df_hnm = df[df['dataset_source'] == 'HNM'].reset_index(drop=True)
    df_fashion = df[df['dataset_source'] == 'FASHION'].reset_index(drop=True)
    
    try:
        df_train_all, df_test_all = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42, 
            stratify=df['dataset_source'] 
        )
    except ValueError:
        print("Warning: Cannot stratify (likely too few samples). Splitting without stratification.")
        df_train_all, df_test_all = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42
        )

    all_results = []
    
    # --- 실험 1: Train HNM, Test FASHION ---
    if not df_hnm.empty and not df_fashion.empty:
        exp1_results = run_experiment(
            "1. Train HNM, Test FASHION (Multimodal)",
            df_hnm, df_fashion, target_cols_dict, num_classes_dict
        )
        all_results.append(exp1_results)
    else:
        print("Skipping Experiment 1: HNM or FASHION data is empty.")

    # --- 실험 2: Train FASHION, Test HNM ---
    if not df_fashion.empty and not df_hnm.empty:
        exp2_results = run_experiment(
            "2. Train FASHION, Test HNM (Multimodal)",
            df_fashion, df_hnm, target_cols_dict, num_classes_dict
        )
        all_results.append(exp2_results)
    else:
        print("Skipping Experiment 2: FASHION or HNM data is empty.")

    # --- 실험 3: Train 80% All, Test 20% All ---
    if not df_train_all.empty and not df_test_all.empty:
        exp3_results = run_experiment(
            "3. Train 80% (Mixed), Test 20% (Mixed) (Multimodal)",
            df_train_all, df_test_all, target_cols_dict, num_classes_dict
        )
        all_results.append(exp3_results)
    else:
        print("Skipping Experiment 3: Split data is empty.")

    # 4. 결과 취합 및 CSV 저장
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        cols_order = ["Experiment", 
                      "Acc (Gender)", "F1 (Gender)", 
                      "Acc (Category)", "F1 (Category)", 
                      "Acc (Usage)", "F1 (Usage)",
                      "Running Time (s)",
                      "Precision (Gender)", "Recall (Gender)",
                      "Precision (Category)", "Recall (Category)",
                      "Precision (Usage)", "Recall (Usage)"]
        
        final_cols = [col for col in cols_order if col in results_df.columns]
        results_df = results_df[final_cols]
        
        results_df.to_csv(RESULTS_CSV_PATH, index=False)
        print(f"\n--- All experiments complete! Results saved to {RESULTS_CSV_PATH} ---")
        print(results_df.to_string())
    else:
        print("No experiments were run. No results to save.")

if __name__ == "__main__":
    main()
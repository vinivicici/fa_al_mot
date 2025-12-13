import pandas as pd
import numpy as np
import os
import glob
import time
from tqdm import tqdm
from PIL import Image

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
RESULTS_CSV_PATH = "" # 최종 결과 CSV 저장 경로

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
                # HNM 파일명 '0108775001.jpg' -> pid '108775' (인덱스 1~6)
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
                # FASHION 파일명 '24085.jpg' -> pid '24085' (확장자 제외)
                pid = f_name_no_ext
                fashion_map[pid] = f_path
        except Exception as e:
            print(f"Skipping FASHION file {f_path}: {e}")
            
    print(f"Found {len(hnm_map)} HNM images and {len(fashion_map)} FASHION images.")
    return hnm_map, fashion_map

def load_and_preprocess_data(csv_path, hnm_map, fashion_map):
    """ 
    CSV를 로드하고, 이미지 맵과 매핑하여 유효한 데이터프레임을 생성합니다.
    또한 'gender', 'category', 'usage' 3개 그룹의 타겟 컬럼을 식별하여 반환합니다.
    """
    print("Loading and preprocessing CSV...")
    
    # CSV 구분자(tab 또는 comma) 자동 감지 시도
    try:
        df = pd.read_csv(csv_path, sep='\t')
        df.columns = df.columns.str.strip()
        if 'product_id' not in df.columns:
            raise ValueError("Tab separator failed, trying comma.")
    except Exception as e:
        print(f"Info: Failed to read with tab (Error: {e})... Trying with comma.")
        try:
            df = pd.read_csv(csv_path) # 기본값 (sep=',')
            df.columns = df.columns.str.strip()
        except Exception as read_e:
            print(f"Fatal Error: Failed to read CSV with both tab and comma. Error: {read_e}")
            return pd.DataFrame(), [], [], []

    # 'product_id' 컬럼 필수 확인
    if 'product_id' not in df.columns:
        print(f"Fatal Error: 'product_id' column not found.")
        print(f"Available columns: {list(df.columns)}")
        raise KeyError("Could not find 'product_id' in CSV columns.")

    df['product_id'] = df['product_id'].astype(str)

    # product_id와 dataset_source를 기반으로 이미지 경로 매핑
    def get_image_path(row):
        pid = row['product_id']
        source = row.get('dataset_source', '') # 'dataset_source'가 없는 경우 대비
        if source == 'HNM':
            return hnm_map.get(pid)
        elif source == 'FASHION':
            return fashion_map.get(pid)
        return None

    df['image_path'] = df.apply(get_image_path, axis=1)
    
    # 실제 이미지 파일이 없는 행 제거
    initial_rows = len(df)
    df = df.dropna(subset=['image_path'])
    final_rows = len(df)
    print(f"Filtered DataFrame: {final_rows} rows remaining (removed {initial_rows - final_rows} rows with no matching image).")

    # 타겟 컬럼을 3개의 그룹으로 분리
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
    이미지 경로와 3가지 태스크(gender, category, usage)의 레이블을 반환하는
    커스텀 PyTorch Dataset 클래스입니다.
    """
    def __init__(self, dataframe, target_cols_dict, transform=None):
        self.df = dataframe
        self.image_paths = dataframe['image_path'].values
        
        # 3개 그룹의 레이블을 Numpy 배열로 미리 추출
        self.gender_labels = dataframe[target_cols_dict['gender']].values.astype(np.float32)
        self.category_labels = dataframe[target_cols_dict['category']].values.astype(np.float32)
        self.usage_labels = dataframe[target_cols_dict['usage']].values.astype(np.float32)
        
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 이미지 로드 실패 시 검은색 이미지로 대체
            print(f"Warning: Error loading image {img_path}, returning black image. Error: {e}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')
            
        if self.transform:
            image = self.transform(image)
            
        # 3개의 레이블을 텐서로 변환
        label_gender = torch.tensor(self.gender_labels[idx], dtype=torch.float32)
        label_category = torch.tensor(self.category_labels[idx], dtype=torch.float32)
        label_usage = torch.tensor(self.usage_labels[idx], dtype=torch.float32)
        
        # 이미지 1개와 (레이블 튜플) 반환
        return image, (label_gender, label_category, label_usage)

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

# --- 4. 모델 정의 (Multi-Task ResNet50) ---

class MultiTaskResNet(nn.Module):
    """
    사전 학습된 ResNet50을 특징 추출기(몸통)로 사용하고,
    3개의 독립적인 분류 헤드(머리)를 가진 멀티태스크 모델.
    """
    def __init__(self, num_classes_dict):
        super(MultiTaskResNet, self).__init__()
        
        # 1. ResNet50(ImageNet 가중치) 로드 및 파라미터 동결 (학습 X)
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in base_model.parameters():
            param.requires_grad = False
            
        # 2. 특징 추출기 (마지막 FC 레이어 제외)
        num_ftrs = base_model.fc.in_features # (e.g., 2048)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        # 3. 3개의 분리된 출력 헤드 (이 부분만 requires_grad=True로 학습됨)
        self.fc_gender = nn.Linear(num_ftrs, num_classes_dict['gender'])
        self.fc_category = nn.Linear(num_ftrs, num_classes_dict['category'])
        self.fc_usage = nn.Linear(num_ftrs, num_classes_dict['usage'])

    def forward(self, x):
        # 1. 공통 특징 추출
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1) # [N, 2048, 1, 1] -> [N, 2048]
        
        # 2. 3개 헤드로 각각 Forward
        out_gender = self.fc_gender(features)
        out_category = self.fc_category(features)
        out_usage = self.fc_usage(features)
        
        return out_gender, out_category, out_usage

# --- 5. 학습 및 평가 함수 ---

def train_epoch(model, dataloader, criterion, optimizer):
    """ 1 에포크 동안 모델을 학습합니다. (멀티태스크 로직 적용) """
    model.train()
    running_loss = 0.0
    
    for inputs, labels_tuple in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(DEVICE)
        
        # 레이블 튜플 분리
        labels_gender, labels_category, labels_usage = (
            labels_tuple[0].to(DEVICE),
            labels_tuple[1].to(DEVICE),
            labels_tuple[2].to(DEVICE)
        )
        
        optimizer.zero_grad()
        
        # 모델에서 3개의 출력 계산
        outputs_gender, outputs_category, outputs_usage = model(inputs)
        
        # 3개 태스크 각각의 손실 계산
        loss_gender = criterion(outputs_gender, labels_gender)
        loss_category = criterion(outputs_category, labels_category)
        loss_usage = criterion(outputs_usage, labels_usage)
        
        # 손실 합산 (가중치 없이 단순 합)
        loss = loss_gender + loss_category + loss_usage
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate_model(model, dataloader):
    """ 테스트 데이터셋으로 모델을 평가합니다. (멀티태스크 로직 적용) """
    model.eval()
    
    # 태스크별 예측/정답 저장 리스트
    all_labels_gender, all_preds_gender = [], []
    all_labels_category, all_preds_category = [], []
    all_labels_usage, all_preds_usage = [], []
    
    with torch.no_grad():
        for inputs, labels_tuple in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            labels_gender, labels_category, labels_usage = labels_tuple
            
            # 모델 출력
            outputs_gender, outputs_category, outputs_usage = model(inputs)
            
            # 1. Gender (Sigmoid + 0.5 임계값)
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
            
    # Numpy 배열로 통합
    all_labels_gender = np.concatenate(all_labels_gender, axis=0)
    all_preds_gender = np.concatenate(all_preds_gender, axis=0)
    all_labels_category = np.concatenate(all_labels_category, axis=0)
    all_preds_category = np.concatenate(all_preds_category, axis=0)
    all_labels_usage = np.concatenate(all_labels_usage, axis=0)
    all_preds_usage = np.concatenate(all_preds_usage, axis=0)
    
    # 메트릭 계산 헬퍼 함수
    def get_metrics(labels, preds):
        accuracy = accuracy_score(labels, preds) # (참고: Exact match ratio)
        precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        return accuracy, precision_macro, recall_macro, f1_macro

    # 태스크별 메트릭 계산
    metrics_gender = get_metrics(all_labels_gender, all_preds_gender)
    metrics_category = get_metrics(all_labels_category, all_preds_category)
    metrics_usage = get_metrics(all_labels_usage, all_preds_usage)
    
    # 결과 딕셔너리 반환
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
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. 모델, 손실함수, 옵티마이저
    model = MultiTaskResNet(num_classes_dict).to(DEVICE)
    
    # 다중 레이블 분류이므로 BCEWithLogitsLoss 사용
    criterion = nn.BCEWithLogitsLoss() 
    
    # 학습 가능한 파라미터(3개의 FC head)만 옵티마이저에 전달
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
    
    # CSV 저장을 위한 최종 결과 딕셔너리
    final_results = {
        "Experiment": exp_name,
        **metrics_dict, # 3개 태스크의 모든 메트릭을 펼쳐서 추가
        "Running Time (s)": running_time
    }
    
    return final_results

# --- 6. 메인 실행 로직 ---

def main():
    # 1. 데이터 준비 (이미지 맵, CSV 로드, 타겟 컬럼 분리)
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
    
    # 데이터 로드 실패 시 종료
    if len(df) == 0:
        print("Fatal Error: No data left after filtering. Exiting.")
        return
    # 타겟 컬럼 분리 실패 시 종료
    if num_classes_dict['gender'] == 0 or num_classes_dict['category'] == 0 or num_classes_dict['usage'] == 0:
        print("Fatal Error: One of the target groups (gender, category, usage) has 0 columns. Check CSV column names. Exiting.")
        return

    # 2. 실험별 데이터 분리
    df_hnm = df[df['dataset_source'] == 'HNM'].reset_index(drop=True)
    df_fashion = df[df['dataset_source'] == 'FASHION'].reset_index(drop=True)
    
    # 3. 전체 데이터셋 (80/20 분할)
    try:
        df_train_all, df_test_all = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42, 
            stratify=df['dataset_source'] # HNM/FASHION 비율 유지
        )
    except ValueError:
        # (데이터가 너무 적어 stratify가 불가능할 경우)
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
            "1. Train HNM, Test FASHION",
            df_hnm, df_fashion, target_cols_dict, num_classes_dict
        )
        all_results.append(exp1_results)
    else:
        print("Skipping Experiment 1: HNM or FASHION data is empty.")

    # --- 실험 2: Train FASHION, Test HNM ---
    if not df_fashion.empty and not df_hnm.empty:
        exp2_results = run_experiment(
            "2. Train FASHION, Test HNM",
            df_fashion, df_hnm, target_cols_dict, num_classes_dict
        )
        all_results.append(exp2_results)
    else:
        print("Skipping Experiment 2: FASHION or HNM data is empty.")

    # --- 실험 3: Train 80% All, Test 20% All ---
    if not df_train_all.empty and not df_test_all.empty:
        exp3_results = run_experiment(
            "3. Train 80% (Mixed), Test 20% (Mixed)",
            df_train_all, df_test_all, target_cols_dict, num_classes_dict
        )
        all_results.append(exp3_results)
    else:
        print("Skipping Experiment 3: Split data is empty.")

    # 4. 결과 취합 및 CSV 저장
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # CSV 컬럼 순서 보기 좋게 정렬
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
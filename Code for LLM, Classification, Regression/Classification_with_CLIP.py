import os
import glob
import time
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# CLIP (OpenCLIP)
import open_clip
import optuna

# -----------------------------
# 1. 설정
# -----------------------------

# [수정 필요] 데이터 경로
CSV_PATH = ""
HNM_IMG_DIR = ""
FASHION_IMG_DIR = ""
RESULTS_CSV_PATH = ""

# 기본 하이퍼파라미터 (Optuna 사용 시 초기값)
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
IMG_SIZE = 224

# CLIP 설정
CLIP_MODEL_NAME = "ViT-B-16"
CLIP_PRETRAINED = "openai"

# Optuna 사용 여부
USE_OPTUNA = True  # False로 두면 그냥 기본 하이퍼파라미터로만 Mixed 80/20 한번 돌림

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# -----------------------------
# 2. 유틸 & 데이터 전처리
# -----------------------------

def build_image_maps():
    """
    HNM / FASHION 디렉토리에서 product_id -> 이미지 경로 매핑 딕셔너리를 생성.
    """
    print("Building image path maps...")
    hnm_map = {}
    fashion_map = {}
    valid_extensions = {".jpg", ".jpeg", ".png"}

    print(f"Scanning HNM directory: {HNM_IMG_DIR}")
    for f_path in tqdm(glob.glob(os.path.join(HNM_IMG_DIR, "*"))):
        try:
            f_name = os.path.basename(f_path)
            ext = os.path.splitext(f_name)[1].lower()
            if ext in valid_extensions and len(f_name) > 7:
                pid = f_name[1:7]  # 예: x123456y -> 123456
                hnm_map[pid] = f_path
        except Exception as e:
            print(f"Skipping HNM file {f_path}: {e}")

    print(f"Scanning FASHION directory: {FASHION_IMG_DIR}")
    for f_path in tqdm(glob.glob(os.path.join(FASHION_IMG_DIR, "*"))):
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
    """
    문자열로 저장된 wrd_emb를 numpy array로 파싱.
    """
    if isinstance(x, np.ndarray):
        return x
    if not isinstance(x, str):
        return None
    try:
        x_clean = re.sub(r"[\n\[\]]", "", x).strip()
        x_clean = re.sub(r"\s+", ",", x_clean)
        x_clean = x_clean.replace(",,", ",")
        return np.fromstring(x_clean, sep=",")
    except Exception as e:
        print(f"Warning: Could not parse embedding string. Error: {e}")
        return None


def load_and_preprocess_data(csv_path, hnm_map, fashion_map):
    """
    CSV 로드 + 이미지 매핑 + wrd_emb 파싱 + 타깃 컬럼 분리.
    """
    print("Loading and preprocessing CSV...")

    # 구분자 추론
    try:
        df = pd.read_csv(csv_path, sep="\t")
        df.columns = df.columns.str.strip()
        if "product_id" not in df.columns:
            raise ValueError("Tab separator failed, trying comma.")
    except Exception as e:
        print(f"Info: Failed to read with tab (Error: {e})... Trying with comma.")
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

    if "product_id" not in df.columns:
        print(f"Available columns: {list(df.columns)}")
        raise KeyError("'product_id' column not found in CSV.")

    if "wrd_emb" not in df.columns:
        print(f"Available columns: {list(df.columns)}")
        raise KeyError("'wrd_emb' column not found. Multi-modal experiment impossible.")

    df["product_id"] = df["product_id"].astype(str)

    # 이미지 경로 매핑
    def get_image_path(row):
        pid = row["product_id"]
        source = row.get("dataset_source", "")
        if source == "HNM":
            return hnm_map.get(pid)
        elif source == "FASHION":
            return fashion_map.get(pid)
        return None

    df["image_path"] = df.apply(get_image_path, axis=1)
    initial_rows = len(df)
    df = df.dropna(subset=["image_path"])
    print(
        f"Filtered DataFrame (image): {len(df)} rows remaining "
        f"(removed {initial_rows - len(df)} rows with no matching image)."
    )

    # wrd_emb 파싱
    print("Parsing word embeddings...")
    df["wrd_emb_parsed"] = df["wrd_emb"].apply(parse_emb)
    initial_rows = len(df)
    df = df.dropna(subset=["wrd_emb_parsed"])
    print(
        f"Filtered DataFrame (wrd_emb): {len(df)} rows remaining "
        f"(removed {initial_rows - len(df)} rows with unparsable 'wrd_emb')."
    )

    # 타깃 컬럼 그룹
    all_cols = df.columns
    gender_cols = [c for c in all_cols if c.startswith("normalized_gender_")]
    category_cols = [c for c in all_cols if c.startswith("normalized_category_")]
    usage_cols = [c for c in all_cols if c.startswith("normalized_usage_")]

    print(f"Identified {len(gender_cols)} gender targets.")
    print(f"Identified {len(category_cols)} category targets.")
    print(f"Identified {len(usage_cols)} usage targets.")

    return df, gender_cols, category_cols, usage_cols


# -----------------------------
# 3. Dataset
# -----------------------------

class FashionImageDataset(Dataset):
    """
    이미지 + wrd_emb + 3개 태스크 레이블을 반환하는 Dataset
    """

    def __init__(self, dataframe, target_cols_dict, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_paths = self.df["image_path"].values

        self.gender_labels = (
            self.df[target_cols_dict["gender"]].values.astype(np.float32)
        )
        self.category_labels = (
            self.df[target_cols_dict["category"]].values.astype(np.float32)
        )
        self.usage_labels = (
            self.df[target_cols_dict["usage"]].values.astype(np.float32)
        )

        self.wrd_embeddings = np.vstack(self.df["wrd_emb_parsed"].values).astype(
            np.float32
        )
        self.text_emb_dim = self.wrd_embeddings.shape[1]
        print(f"Dataset initialized with text embedding dimension: {self.text_emb_dim}")

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Error loading image {img_path}, using black image. ({e})")
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color="black")

        if self.transform:
            image = self.transform(image)

        wrd_emb = torch.tensor(self.wrd_embeddings[idx], dtype=torch.float32)

        label_gender = torch.tensor(self.gender_labels[idx], dtype=torch.float32)
        label_category = torch.tensor(self.category_labels[idx], dtype=torch.float32)
        label_usage = torch.tensor(self.usage_labels[idx], dtype=torch.float32)

        return image, wrd_emb, (label_gender, label_category, label_usage)


# -----------------------------
# 4. Transform (CLIP 스타일)
# -----------------------------

# CLIP에서 사용하는 정규화 값 (OpenAI CLIP 기준)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    ),
}


# -----------------------------
# 5. Multi-modal CLIP 모델
# -----------------------------

class MultiModalCLIP(nn.Module):
    """
    Frozen CLIP ViT-B/16 (image encoder) + text embedding concat + multi-task head
    """

    def __init__(
        self,
        num_classes_dict,
        text_emb_dim,
        fusion_hidden_dim=1024,
        dropout=0.2,
        clip_model_name=CLIP_MODEL_NAME,
        clip_pretrained=CLIP_PRETRAINED,
    ):
        super().__init__()

        # 1. CLIP 모델 로드
        clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        self.clip_model = clip_model

        # 2. CLIP 파라미터 동결 + 항상 eval 모드 유지
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        # open_clip에서 visual output dim
        img_feature_dim = self.clip_model.visual.output_dim  # ViT-B/16: 512
        self.img_feature_dim = img_feature_dim
        self.text_emb_dim = text_emb_dim

        print(
            f"MultiModalCLIP init: img_feature_dim={img_feature_dim}, "
            f"text_emb_dim={text_emb_dim}, fusion_hidden_dim={fusion_hidden_dim}"
        )

        # 3. LayerNorm으로 이미지/텍스트 피처 스케일 정리
        self.img_ln = nn.LayerNorm(img_feature_dim)
        self.txt_ln = nn.LayerNorm(text_emb_dim)

        # 4. Fusion MLP
        combined_dim = img_feature_dim + text_emb_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
        )

        # 5. 3개 태스크 헤드
        self.fc_gender = nn.Linear(fusion_hidden_dim, num_classes_dict["gender"])
        self.fc_category = nn.Linear(fusion_hidden_dim, num_classes_dict["category"])
        self.fc_usage = nn.Linear(fusion_hidden_dim, num_classes_dict["usage"])

    def encode_image(self, x_img):
        # CLIP의 encode_image 사용 (pre-trained visual encoder)
        img_feat = self.clip_model.encode_image(x_img)  # [N, D]
        return img_feat

    def forward(self, x_img, x_text):
        # CLIP는 항상 eval 동작하도록 보장
        self.clip_model.eval()

        img_features = self.encode_image(x_img)  # [N, D_img]
        img_features = self.img_ln(img_features)

        txt_features = self.txt_ln(x_text)  # [N, D_txt]

        combined = torch.cat([img_features, txt_features], dim=1)
        fused = self.fusion(combined)

        out_gender = self.fc_gender(fused)
        out_category = self.fc_category(fused)
        out_usage = self.fc_usage(fused)

        return out_gender, out_category, out_usage

    def train(self, mode: bool = True):
        """
        train()/eval() 호출 시에도 clip_model은 항상 eval 모드로 유지하고,
        fusion + heads만 train/eval 모드 전환.
        """
        super().train(mode)
        self.clip_model.eval()  # 항상 고정
        return self


# -----------------------------
# 6. 학습 & 평가 함수
# -----------------------------

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for img_inputs, text_inputs, labels_tuple in tqdm(dataloader, desc="Training"):
        img_inputs = img_inputs.to(DEVICE)
        text_inputs = text_inputs.to(DEVICE)
        labels_gender = labels_tuple[0].to(DEVICE)
        labels_category = labels_tuple[1].to(DEVICE)
        labels_usage = labels_tuple[2].to(DEVICE)

        optimizer.zero_grad()

        outputs_gender, outputs_category, outputs_usage = model(
            img_inputs, text_inputs
        )

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
    model.eval()

    all_labels_gender, all_preds_gender = [], []
    all_labels_category, all_preds_category = [], []
    all_labels_usage, all_preds_usage = [], []

    with torch.no_grad():
        for img_inputs, text_inputs, labels_tuple in tqdm(
            dataloader, desc="Evaluating"
        ):
            img_inputs = img_inputs.to(DEVICE)
            text_inputs = text_inputs.to(DEVICE)
            labels_gender = labels_tuple[0]
            labels_category = labels_tuple[1]
            labels_usage = labels_tuple[2]

            outputs_gender, outputs_category, outputs_usage = model(
                img_inputs, text_inputs
            )

            # sigmoid + threshold
            preds_gender = (torch.sigmoid(outputs_gender) > 0.5).float()
            preds_category = (torch.sigmoid(outputs_category) > 0.5).float()
            preds_usage = (torch.sigmoid(outputs_usage) > 0.5).float()

            all_labels_gender.append(labels_gender.cpu().numpy())
            all_preds_gender.append(preds_gender.cpu().numpy())
            all_labels_category.append(labels_category.cpu().numpy())
            all_preds_category.append(preds_category.cpu().numpy())
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
        precision_macro = precision_score(
            labels, preds, average="macro", zero_division=0
        )
        recall_macro = recall_score(labels, preds, average="macro", zero_division=0)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        return accuracy, precision_macro, recall_macro, f1_macro

    metrics_gender = get_metrics(all_labels_gender, all_preds_gender)
    metrics_category = get_metrics(all_labels_category, all_preds_category)
    metrics_usage = get_metrics(all_labels_usage, all_preds_usage)

    results = {
        "Acc (Gender)": metrics_gender[0],
        "F1 (Gender)": metrics_gender[3],
        "Acc (Category)": metrics_category[0],
        "F1 (Category)": metrics_category[3],
        "Acc (Usage)": metrics_usage[0],
        "F1 (Usage)": metrics_usage[3],
        "Precision (Gender)": metrics_gender[1],
        "Recall (Gender)": metrics_gender[2],
        "Precision (Category)": metrics_category[1],
        "Recall (Category)": metrics_category[2],
        "Precision (Usage)": metrics_usage[1],
        "Recall (Usage)": metrics_usage[2],
    }

    return results


# -----------------------------
# 7. 한 실험 실행 (Mixed용)
# -----------------------------

def run_experiment(
    exp_name,
    df_train,
    df_test,
    target_cols_dict,
    num_classes_dict,
    lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    fusion_hidden_dim=1024,
    dropout=0.2,
):
    """
    하나의 실험 시나리오 실행.
    hyperparameter(lr, batch_size, fusion_hidden_dim, dropout)는
    Optuna에서 override 가능하도록 인자로 둠.
    """
    print(f"\n--- Starting Experiment: {exp_name} ---")
    start_time = time.time()

    train_dataset = FashionImageDataset(
        df_train, target_cols_dict, transform=data_transforms["train"]
    )
    test_dataset = FashionImageDataset(
        df_test, target_cols_dict, transform=data_transforms["test"]
    )

    text_emb_dim = train_dataset.text_emb_dim

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 모델 / 손실 / 옵티마이저
    model = MultiModalCLIP(
        num_classes_dict,
        text_emb_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        dropout=dropout,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    # 학습
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1} Train Loss (Combined): {train_loss:.4f}")

    # 평가
    metrics_dict = evaluate_model(model, test_loader)

    end_time = time.time()
    running_time = end_time - start_time

    print(f"--- Experiment Finished: {exp_name} ---")
    print(
        f"Gender Metrics:   Acc={metrics_dict['Acc (Gender)']:.4f}, "
        f"F1={metrics_dict['F1 (Gender)']:.4f}"
    )
    print(
        f"Category Metrics: Acc={metrics_dict['Acc (Category)']:.4f}, "
        f"F1={metrics_dict['F1 (Category)']:.4f}"
    )
    print(
        f"Usage Metrics:    Acc={metrics_dict['Acc (Usage)']:.4f}, "
        f"F1={metrics_dict['F1 (Usage)']:.4f}"
    )
    print(f"Total Running Time: {running_time:.2f} seconds")

    final_results = {
        "Experiment": exp_name,
        **metrics_dict,
        "Running Time (s)": running_time,
    }
    return final_results


# -----------------------------
# 8. Optuna 튜닝 (Mixed 80% 내부에서만)
# -----------------------------

def tune_with_optuna(df_train_all, target_cols_dict, num_classes_dict, n_trials=2):
    """
    Mixed 80% 중에서만 train/valid로 나눠 Optuna 튜닝.
    """

    def objective(trial):
        # 1) 하이퍼파라미터 샘플링
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        fusion_hidden_dim = trial.suggest_categorical(
            "fusion_hidden_dim", [512, 1024, 2048]
        )
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        # 2) train / valid split (df_train_all에서 다시 8:2)
        df_train, df_valid = train_test_split(
            df_train_all,
            test_size=0.2,
            random_state=trial.number,  # trial마다 약간 다르게
        )

        train_dataset = FashionImageDataset(
            df_train, target_cols_dict, transform=data_transforms["train"]
        )
        valid_dataset = FashionImageDataset(
            df_valid, target_cols_dict, transform=data_transforms["test"]
        )

        text_emb_dim = train_dataset.text_emb_dim

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        model = MultiModalCLIP(
            num_classes_dict,
            text_emb_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            dropout=dropout,
        ).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=WEIGHT_DECAY,
        )

        best_avg_f1 = -1.0

        # 3) 몇 epoch만 빠르게 돌리기 (튜닝용)
        for epoch in range(5):
            _ = train_epoch(model, train_loader, criterion, optimizer)
            metrics = evaluate_model(model, valid_loader)

            avg_f1 = (
                metrics["F1 (Gender)"]
                + metrics["F1 (Category)"]
                + metrics["F1 (Usage)"]
            ) / 3.0

            best_avg_f1 = max(best_avg_f1, avg_f1)

            trial.report(avg_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_avg_f1

    # 4) Optuna Study 실행
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\n[Optuna] Best value (avg F1):", study.best_value)
    print("[Optuna] Best params:", study.best_params)

    return study.best_params


# -----------------------------
# 9. 메인: Mixed 80/20 한 번만
# -----------------------------

def main():
    # 1) 데이터 로딩 & 전처리
    hnm_map, fashion_map = build_image_maps()
    df, gender_cols, category_cols, usage_cols = load_and_preprocess_data(
        CSV_PATH, hnm_map, fashion_map
    )

    target_cols_dict = {
        "gender": gender_cols,
        "category": category_cols,
        "usage": usage_cols,
    }
    num_classes_dict = {
        "gender": len(gender_cols),
        "category": len(category_cols),
        "usage": len(usage_cols),
    }

    if len(df) == 0:
        print("Fatal Error: No data left after filtering. Exiting.")
        return
    if (
        num_classes_dict["gender"] == 0
        or num_classes_dict["category"] == 0
        or num_classes_dict["usage"] == 0
    ):
        print("Fatal Error: One of the target groups has 0 columns.")
        return

    # 2) Mixed 80/20 split (dataset_source 비율 유지)
    try:
        df_train_all, df_test_all = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["dataset_source"],
        )
    except ValueError:
        print(
            "Warning: Cannot stratify by dataset_source. Splitting without stratification."
        )
        df_train_all, df_test_all = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
        )

    # 3) Optuna로 튜닝
    if USE_OPTUNA:
        best_params = tune_with_optuna(
            df_train_all, target_cols_dict, num_classes_dict, n_trials=3
        )
        print("\n[Optuna] Best params:", best_params)

        lr = best_params["lr"]
        batch_size = best_params["batch_size"]
        fusion_hidden_dim = best_params["fusion_hidden_dim"]
        dropout = best_params["dropout"]
    else:
        lr = LEARNING_RATE
        batch_size = BATCH_SIZE
        fusion_hidden_dim = 1024
        dropout = 0.2

    # 4) 최종 Mixed 80/20 실험 한 번만
    results = run_experiment(
        "3. Train 80% (Mixed), Test 20% (Mixed) (CLIP multimodal)",
        df_train_all,
        df_test_all,
        target_cols_dict,
        num_classes_dict,
        lr=lr,
        batch_size=batch_size,
        fusion_hidden_dim=fusion_hidden_dim,
        dropout=dropout,
    )

    # 5) 결과 저장
    results_df = pd.DataFrame([results])
    cols_order = [
        "Experiment",
        "Acc (Gender)",
        "F1 (Gender)",
        "Acc (Category)",
        "F1 (Category)",
        "Acc (Usage)",
        "F1 (Usage)",
        "Running Time (s)",
        "Precision (Gender)",
        "Recall (Gender)",
        "Precision (Category)",
        "Recall (Category)",
        "Precision (Usage)",
        "Recall (Usage)",
    ]
    final_cols = [c for c in cols_order if c in results_df.columns]
    results_df = results_df[final_cols]

    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(
        f"\n--- Mixed 80/20 experiment complete! Results saved to {RESULTS_CSV_PATH} ---"
    )
    print(results_df.to_string())


if __name__ == "__main__":
    main()

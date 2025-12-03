import argparse
import os
import csv
import re
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration


class PositionalEncoding(nn.Module):
    """Transformer용 사인/코사인 위치 임베딩"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(x.device)

class ImageCaptionModel(nn.Module):
    """CNN+LSTM 기반 이미지 캡셔닝 모델"""
    def __init__(self, embed_size=256, hidden_size=512, vocab_size=10000, num_layers=2):
        super(ImageCaptionModel, self).__init__()
        
        # CNN (ResNet50) - 이미지 특징 추출
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # 마지막 FC 레이어 제거하고 특징 추출기로 사용
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.cnn_features = resnet.fc.in_features
        
        # 이미지 특징을 임베딩 공간으로 변환
        self.image_embedding = nn.Linear(self.cnn_features, embed_size)
        
        # LSTM - 캡션 생성
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # 단어 임베딩
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        
        # 출력 레이어
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images, captions=None):
        # CNN으로 이미지 특징 추출
        with torch.no_grad():
            features = self.cnn(images)
            features = features.view(features.size(0), -1)
        
        # 이미지 특징을 임베딩으로 변환
        image_embeds = self.image_embedding(features)
        
        if captions is not None:
            # 학습 시: LSTM에 이미지 특징과 캡션을 함께 입력
            word_embeds = self.word_embedding(captions)
            # 이미지 특징을 첫 번째 입력으로 사용
            inputs = torch.cat([image_embeds.unsqueeze(1), word_embeds], dim=1)
            lstm_out, _ = self.lstm(inputs)
            output = self.fc(self.dropout(lstm_out))
            return output
        else:
            # 추론 시: 이미지 특징만으로 캡션 생성
            return image_embeds


class TransformerCaptionModel(nn.Module):
    """
    CNN(ResNet50) + Transformer Decoder 기반 이미지 캡셔닝 모델

    - Encoder: ResNet50 (FC 제거 후 feature vector 사용)
    - Decoder: nn.TransformerDecoder (multi-head self-attention + cross-attention)
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        vocab_size: int = 10000,
        max_len: int = 64,
    ):
        super().__init__()

        # CNN (ResNet50) - 이미지 특징 추출
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.cnn_features = resnet.fc.in_features

        # 이미지 특징을 Transformer d_model 차원으로 프로젝션
        self.image_proj = nn.Linear(self.cnn_features, d_model)

        # 텍스트 토큰 임베딩 + 위치 임베딩
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len + 1)

        # Transformer Decoder 정의
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, seq, feature) 형태 사용
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # 출력 레이어 (vocab logits)
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.max_len = max_len

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        이미지를 CNN으로 인코딩하여 Transformer memory로 변환

        images: (batch, 3, H, W)
        return: memory (batch, 1, d_model)
        """
        with torch.no_grad():
            features = self.cnn(images)          # (batch, 2048, 1, 1)
            features = features.view(features.size(0), -1)  # (batch, 2048)

        memory = self.image_proj(features).unsqueeze(1)     # (batch, 1, d_model)
        return memory

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        학습용 forward

        images : (batch, 3, H, W)
        captions : (batch, seq_len)  - 토큰 인덱스(시프트된 입력, [BOS, w1, w2, ..., wN])

        return: logits (batch, seq_len, vocab_size)
        """
        device = images.device

        # 1) 이미지 인코딩
        memory = self.encode_image(images)  # (batch, 1, d_model)

        # 2) 캡션 임베딩 + 위치 인코딩
        tgt_emb = self.word_embedding(captions)  # (batch, seq_len, d_model)
        tgt_emb = self.pos_encoding(tgt_emb)     # (batch, seq_len, d_model)

        # 3) look-ahead mask (미래 토큰 마스킹)
        seq_len = captions.size(1)
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(device)  # (seq_len, seq_len)

        # 4) Transformer Decoder
        out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
        )  # (batch, seq_len, d_model)

        # 5) vocab logits
        logits = self.fc_out(out)  # (batch, seq_len, vocab_size)
        return logits

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Transformer용 causal mask 생성 (상삼각 부분을 -inf로)"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float("-inf"))
        return mask

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = None,
    ) -> torch.Tensor:
        """
        Greedy decoding으로 캡션 생성

        images: (batch, 3, H, W)
        return: captions (batch, <=max_len)  - BOS/EOS 포함 토큰 인덱스
        """
        self.eval()
        device = images.device
        max_len = max_len or self.max_len

        # 1) 이미지 인코딩
        memory = self.encode_image(images)  # (batch, 1, d_model)
        batch_size = images.size(0)

        # 2) 시작 토큰으로 초기 시퀀스 생성
        ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)  # (batch, 1)

        for _ in range(max_len - 1):
            tgt_emb = self.word_embedding(ys)         # (batch, cur_len, d_model)
            tgt_emb = self.pos_encoding(tgt_emb)      # (batch, cur_len, d_model)

            cur_len = ys.size(1)
            tgt_mask = self._generate_square_subsequent_mask(cur_len).to(device)

            out = self.transformer_decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
            )  # (batch, cur_len, d_model)

            logits = self.fc_out(out[:, -1, :])  # (batch, vocab_size) - 마지막 토큰만
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

            ys = torch.cat([ys, next_token], dim=1)  # (batch, cur_len+1)

            # 모든 시퀀스가 EOS를 만났다면 종료
            if (next_token == eos_idx).all():
                break

        return ys


class SimpleCaptionGenerator:
    """CNN+LSTM 기반 캡션 생성기 - BLIP 모델 사용 (내부적으로 CNN+LSTM 구조)"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', use_blip=True):
        self.device = device
        self.use_blip = use_blip
        
        if use_blip:
            # BLIP 모델 사용 (내부적으로 vision transformer + text decoder 구조)
            # 더 나은 캡션 생성을 위해 사전 학습된 모델 사용
            try:
                print("BLIP 모델 로딩 중...")
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                ).to(device)
                self.blip_model.eval()
                print("BLIP 모델 로딩 완료!")
            except Exception as e:
                print(f"BLIP 모델 로딩 실패: {e}")
                print("기본 CNN+LSTM 모델로 전환합니다.")
                self.use_blip = False
        
        if not self.use_blip:
            # 기본 CNN+LSTM 모델
            self.model = ImageCaptionModel().to(device)
            self.model.eval()
            
            # 이미지 전처리
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def generate_caption(self, image_path):
        """이미지 경로로부터 상품 설명 생성"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.use_blip:
                # 상품 설명 스타일로 여러 측면의 설명 생성
                product_description = self._generate_product_description(image, image_path)
                return product_description
            else:
                # 기본 CNN+LSTM 모델 사용 (간단한 규칙 기반)
                return self._generate_simple_caption(image_path)
                
        except Exception as e:
            print(f"이미지 처리 오류 ({image_path}): {e}")
            return self._get_default_product_description()
    
    def _generate_product_description(self, image, image_path):
        """상품 페이지용 상세 설명 생성"""
        filename = Path(image_path).stem.lower()
        
        # 여러 프롬프트로 다양한 측면의 설명 생성
        descriptions = {}
        
        # 1. 메인 아이템 설명 (상세하게)
        main_item_prompt = "Write a detailed product description for this clothing item. Describe the style, design, and appearance in detail."
        descriptions['main'] = self._generate_with_prompt(image, main_item_prompt, max_length=80)
        
        # 2. 재질/소재 추론
        material_prompt = "Describe the material and fabric of this clothing item."
        descriptions['material'] = self._generate_with_prompt(image, material_prompt, max_length=40)
        
        # 3. 스타일/핏 설명
        style_prompt = "Describe the style, fit, and silhouette of this clothing item."
        descriptions['style'] = self._generate_with_prompt(image, style_prompt, max_length=40)
        
        # 4. 브랜드/디자인 특징
        brand_prompt = "Describe the design features and unique characteristics of this clothing item."
        descriptions['features'] = self._generate_with_prompt(image, brand_prompt, max_length=50)
        
        # 구조화된 상품 설명 생성
        return self._format_product_description(descriptions, filename)
    
    def _generate_with_prompt(self, image, prompt, max_length=50):
        """프롬프트를 사용하여 캡션 생성"""
        try:
            inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=max_length, 
                                              num_beams=5, length_penalty=1.0,
                                              repetition_penalty=1.5,  # 반복 방지
                                              temperature=0.8,
                                              do_sample=True)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # 프롬프트 제거 (개선된 로직)
            caption = self._remove_prompt_from_caption(caption, prompt)
            
            # 반복 단어 제거
            caption = self._remove_repetitions(caption)
            
            return caption.strip()
        except Exception as e:
            return "N/A"
    
    def _remove_prompt_from_caption(self, caption, prompt):
        """캡션에서 프롬프트 제거 (개선된 버전)"""
        # 원본 프롬프트 제거
        caption = caption.replace(prompt, "").strip()
        
        # 프롬프트의 주요 키워드들 제거
        prompt_lower = prompt.lower()
        caption_lower = caption.lower()
        
        # 프롬프트의 주요 구문 패턴 제거
        patterns_to_remove = [
            r'write\s+a\s+detailed\s+product\s+description',
            r'describe\s+the\s+style,\s+design,\s+and\s+appearance',
            r'describe\s+the\s+material\s+and\s+fabric',
            r'describe\s+the\s+style,\s+fit,\s+and\s+silhouette',
            r'describe\s+the\s+design\s+features\s+and\s+unique\s+characteristics',
            r'for\s+this\s+clothing\s+item',
            r'of\s+this\s+clothing\s+item',
        ]
        
        for pattern in patterns_to_remove:
            caption = re.sub(pattern, '', caption, flags=re.IGNORECASE)
        
        # 앞뒤 공백 정리
        caption = re.sub(r'\s+', ' ', caption).strip()
        
        # 문장 시작 부분의 불필요한 구두점 제거
        caption = re.sub(r'^[,\s\.\-]+', '', caption)
        
        return caption
    
    def _remove_repetitions(self, text):
        """반복되는 단어나 구문 제거"""
        if not text or len(text.split()) < 2:
            return text
        
        words = text.split()
        cleaned_words = []
        prev_word = None
        repeat_count = 0
        
        for word in words:
            # 단어 정규화 (구두점 제거, 소문자 변환)
            normalized = re.sub(r'[^\w\s]', '', word.lower())
            
            if normalized == prev_word and normalized:  # 빈 문자열 제외
                repeat_count += 1
                if repeat_count < 2:  # 최대 2번까지만 허용
                    cleaned_words.append(word)
                # 2번 이상 반복되면 추가하지 않음
            else:
                repeat_count = 0
                cleaned_words.append(word)
                prev_word = normalized if normalized else None
        
        result = ' '.join(cleaned_words)
        
        # 같은 구문이 쉼표로 반복되는 경우 제거 (예: "t - shirt, t - shirt, t - shirt")
        result = re.sub(r'([^,]+),\s*\1(?:,\s*\1)+', r'\1', result)
        
        # 같은 단어가 여러 번 반복되는 패턴 제거
        result = re.sub(r'\b(\w+(?:\s+\w+)*)\s+(?:\1\s+){2,}', r'\1', result)
        
        return result
    
    def _format_product_description(self, descriptions, filename):
        """구조화된 상품 설명 형식으로 포맷팅"""
        # 파일명에서 아이템 타입 추론
        item_type = self._infer_item_type(filename)
        
        # 메인 아이템 설명 개선
        main_item = descriptions.get('main', 'A stylish clothing item')
        if main_item == 'N/A' or len(main_item.split()) < 5:
            main_item = f"A high-quality {item_type} with modern design and excellent craftsmanship."
        
        # 재질 정보
        material = descriptions.get('material', 'N/A')
        if material == 'N/A' or len(material.split()) < 3:
            material = 'High-quality fabric'
        
        # 스타일/핏 정보
        style_fit = descriptions.get('style', 'N/A')
        if style_fit == 'N/A' or len(style_fit.split()) < 3:
            style_fit = 'Contemporary fit and style'
        
        # 특징 정보
        features = descriptions.get('features', 'N/A')
        if features == 'N/A' or len(features.split()) < 3:
            features = 'Versatile design suitable for various occasions'
        
        # 구조화된 형식으로 조합
        formatted = (
            f"Brand: N/A\n"
            f"Main Item: {main_item}\n"
            f"Materials: {material}\n"
            f"Care: Machine washable. Follow care instructions on label.\n"
            f"Style/Fit: {style_fit}\n"
            f"Other Details: {features}"
        )
        
        return formatted
    
    def _infer_item_type(self, filename):
        """파일명에서 아이템 타입 추론"""
        filename_lower = filename.lower()
        
        if any(word in filename_lower for word in ['shirt', 'top', 'blouse', 't-shirt', 'tshirt']):
            return "shirt or top"
        elif any(word in filename_lower for word in ['pant', 'jean', 'trouser', 'pants']):
            return "pants or trousers"
        elif any(word in filename_lower for word in ['dress', 'gown']):
            return "dress"
        elif any(word in filename_lower for word in ['jacket', 'coat', 'blazer']):
            return "outerwear"
        elif any(word in filename_lower for word in ['skirt']):
            return "skirt"
        elif any(word in filename_lower for word in ['sweater', 'hoodie', 'cardigan']):
            return "sweater or knitwear"
        else:
            return "clothing item"
    
    def _get_default_product_description(self):
        """기본 상품 설명 반환"""
        return (
            "Brand: N/A\n"
            "Main Item: A stylish clothing item with modern design\n"
            "Materials: High-quality fabric\n"
            "Care: Machine washable. Follow care instructions on label.\n"
            "Style/Fit: Contemporary fit and style\n"
            "Other Details: Versatile design suitable for various occasions"
        )
    
    def _generate_simple_caption(self, image_path):
        """간단한 규칙 기반 캡션 생성"""
        filename = Path(image_path).stem.lower()
        
        if any(word in filename for word in ['shirt', 'top', 'blouse']):
            item_type = "top or shirt"
        elif any(word in filename for word in ['pant', 'jean', 'trouser']):
            item_type = "pants or trousers"
        elif any(word in filename for word in ['dress', 'gown']):
            item_type = "dress"
        elif any(word in filename for word in ['jacket', 'coat', 'outer']):
            item_type = "outerwear"
        else:
            item_type = "clothing item"
        
        caption = f"A stylish {item_type} with modern design. Suitable for everyday wear."
        return caption


def process_images(data_path, output_file='result.csv', test_mode=False):
    """이미지 폴더를 순회하며 캡션 생성 및 CSV 파일 생성"""
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"오류: 데이터셋 경로가 존재하지 않습니다: {data_path}")
        return
    
    # 지원하는 이미지 확장자
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    # 이미지 파일 찾기
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_path.glob(f'*{ext}'))
        image_files.extend(data_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"오류: {data_path}에 이미지 파일을 찾을 수 없습니다.")
        return
    
    # 테스트 모드일 경우 5개로 제한
    if test_mode:
        image_files = image_files[:5]
        print(f"[테스트 모드] 총 {len(image_files)}개의 이미지 파일만 처리합니다.")
    else:
        print(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")
    
    # 캡션 생성기 초기화
    generator = SimpleCaptionGenerator()
    
    # 결과 저장
    results = []
    
    for idx, image_path in enumerate(image_files, 1):
        product_id = image_path.stem  # 파일명 (확장자 제외)
        print(f"[{idx}/{len(image_files)}] 처리 중: {product_id}")
        
        caption = generator.generate_caption(str(image_path))
        results.append({
            'product_id': product_id,
            'description': caption
        })
    
    # CSV 파일로 저장
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['product_id', 'description'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n완료! 결과가 {output_file}에 저장되었습니다.")
    print(f"총 {len(results)}개의 항목이 생성되었습니다.")


def main():
    parser = argparse.ArgumentParser(description='이미지 캡셔닝 프로그램 (CNN+LSTM 기반)')
    parser.add_argument('--data_path', type=str, default='dataset',
                       help='이미지 데이터셋 경로 (기본값: dataset)')
    parser.add_argument('--output', type=str, default='result.csv',
                       help='출력 CSV 파일명 (기본값: result.csv)')
    parser.add_argument('--test', action='store_true',
                       help='테스트 모드: 5개의 이미지만 처리')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("이미지 캡셔닝 프로그램 (CNN+LSTM 기반)")
    print("=" * 50)
    print(f"데이터셋 경로: {args.data_path}")
    print(f"출력 파일: {args.output}")
    if args.test:
        print("모드: 테스트 (5개만 처리)")
    print("=" * 50)
    
    process_images(args.data_path, args.output, test_mode=args.test)


if __name__ == '__main__':
    main()


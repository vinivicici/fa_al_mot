"""
이미지 캡셔닝 스크립트
CNN과 RNN을 활용하여 이미지 폴더의 이미지들에 대한 캡션을 생성하고 CSV 파일로 저장합니다.
"""

import os
# OpenMP 라이브러리 충돌 해결 (가장 먼저 설정)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # 추가 안정성

import csv
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
from pathlib import Path


class ImageCaptioningModel(nn.Module):
    """CNN + RNN 기반 이미지 캡셔닝 모델"""
    
    def __init__(self, embed_size=256, hidden_size=512, vocab_size=10000, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        
        # CNN: ResNet을 특징 추출기로 사용
        resnet = models.resnet50(pretrained=True)
        # 마지막 FC 레이어 제거
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.cnn_features = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
        # RNN: LSTM을 사용하여 캡션 생성
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # 출력 레이어
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        
    def forward(self, images, captions):
        """Forward pass"""
        # CNN으로 이미지 특징 추출
        features = self.cnn(images)
        features = features.reshape(features.size(0), -1)
        features = self.cnn_features(features)
        features = self.bn(features)
        
        # RNN으로 캡션 생성
        embeddings = self.embed(captions)
        features = features.unsqueeze(1)
        hiddens, _ = self.rnn(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class SimpleImageCaptioner:
    """간단한 이미지 캡셔닝 클래스 (사전 학습된 모델 사용)"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 중인 디바이스: {self.device}")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # BLIP 모델 로드 시도 (더 나은 품질을 위해 large 모델 시도)
        self.use_blip = False
        self.model_name = None
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            print("BLIP 모델을 로드하는 중...")
            
            # 먼저 large 모델 시도, 실패하면 base 모델 사용
            try:
                print("BLIP-large 모델 로드 시도 중...")
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)
                self.model_name = "large"
            except:
                print("BLIP-large 로드 실패, base 모델 사용...")
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
                self.model_name = "base"
            
            self.model.eval()  # 평가 모드로 설정
            self.use_blip = True
            print(f"BLIP-{self.model_name} 모델 로드 완료!")
        except ImportError as e:
            print(f"transformers 라이브러리가 없습니다: {e}")
            print("간단한 설명 생성 모드를 사용합니다.")
        except Exception as e:
            print(f"BLIP 모델 로드 중 오류 발생: {e}")
            print("간단한 설명 생성 모드로 전환합니다.")
        
        if not self.use_blip:
            # ResNet으로 특징 추출 (fallback)
            resnet = models.resnet50(pretrained=True)
            self.cnn = nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
            self.cnn.eval()
    
    def generate_caption(self, image_path):
        """의류 이미지에 대한 상세한 설명 생성"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.use_blip:
                # BLIP 모델 사용 (내부적으로 CNN + Transformer 사용)
                # 의류 설명을 위한 프롬프트 추가
                prompt = "Describe this clothing item in detail, including the item type, materials, style, fit, and features."
                
                with torch.no_grad():
                    inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
                    # 더 나은 캡션 생성을 위한 파라미터 설정
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=120,  # 의류 설명은 더 길 수 있음
                        min_length=20,  # 최소 길이 설정
                        num_beams=5,  # beam search 사용
                        repetition_penalty=1.5,  # 반복 방지
                        length_penalty=1.2,  # 더 긴 캡션 선호
                        do_sample=False,
                        early_stopping=True
                    )
                    caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                    
                    # 프롬프트 제거 (프롬프트가 결과에 포함될 수 있음)
                    if prompt.lower() in caption.lower():
                        caption = caption.replace(prompt, "").strip()
                    
                    # 빈 캡션이나 너무 짧은 경우 재시도 (프롬프트 없이)
                    if not caption or len(caption.strip()) < 10:
                        inputs = self.processor(image, return_tensors="pt").to(self.device)
                        generated_ids = self.model.generate(
                            **inputs,
                            max_length=120,
                            num_beams=3,
                            repetition_penalty=1.3,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9
                        )
                        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                    
                    # 여전히 문제가 있으면 fallback
                    if not caption or len(caption.strip()) < 5:
                        caption = self._generate_simple_caption(None, image_path)
                    else:
                        # 구조화된 형식으로 변환
                        caption = self._format_organized_description(caption)
            else:
                # 간단한 설명 생성 (이미지 특징 기반)
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.cnn(image_tensor)
                    features = features.squeeze().cpu().numpy()
                
                # 특징을 기반으로 간단한 설명 생성
                caption = self._generate_simple_caption(features, image_path)
            
            return caption
        except Exception as e:
            print(f"이미지 처리 중 오류 발생 ({image_path}): {str(e)}")
            import traceback
            traceback.print_exc()
            return "이미지를 처리할 수 없습니다."
    
    def _format_organized_description(self, caption):
        """일반 캡션을 구조화된 형식(Brand, Main Item, Materials 등)으로 변환"""
        import re
        
        # 기본값
        brand = "N/A"
        main_item = "N/A"
        materials = "N/A"
        care = "N/A"
        style_fit = "N/A"
        other_details = "N/A"
        
        # 캡션을 소문자로 변환하여 분석
        caption_lower = caption.lower()
        
        # Main Item 추출 (의류 타입 찾기)
        clothing_types = [
            'top', 'shirt', 't-shirt', 'blouse', 'sweater', 'hoodie', 'jacket', 'coat',
            'dress', 'skirt', 'pants', 'trousers', 'jeans', 'shorts', 'leggings',
            'bra', 'underwear', 'lingerie', 'socks', 'stockings', 'tights',
            'shoes', 'sneakers', 'boots', 'sandals', 'heels',
            'hat', 'cap', 'scarf', 'gloves', 'bag', 'accessories'
        ]
        
        for item_type in clothing_types:
            if item_type in caption_lower:
                # 원본 캡션에서 정확한 단어 찾기
                pattern = r'\b' + re.escape(item_type) + r'\b'
                matches = re.findall(pattern, caption_lower, re.IGNORECASE)
                if matches:
                    # 원본 캡션에서 대소문자 유지하며 찾기
                    for word in caption.split():
                        if word.lower() == item_type:
                            main_item = word.capitalize()
                            break
                    if main_item == "N/A":
                        main_item = item_type.capitalize()
                break
        
        # Materials 추출 (재질 키워드 찾기)
        material_keywords = {
            'cotton': 'Cotton',
            'polyester': 'Polyester',
            'nylon': 'Nylon',
            'wool': 'Wool',
            'silk': 'Silk',
            'linen': 'Linen',
            'denim': 'Denim',
            'leather': 'Leather',
            'jersey': 'Jersey',
            'microfibre': 'Microfibre',
            'spandex': 'Spandex',
            'elastane': 'Elastane',
            'viscose': 'Viscose',
            'rayon': 'Rayon',
            'cashmere': 'Cashmere',
            'organic cotton': 'Organic cotton',
            'soft cotton': 'Soft cotton',
            'stretch': 'Stretch',
            'brushed': 'Brushed'
        }
        
        found_materials = []
        for keyword, label in material_keywords.items():
            if keyword in caption_lower:
                found_materials.append(label)
        
        if found_materials:
            materials = ', '.join(found_materials)
        
        # Style/Fit 추출 (스타일 관련 키워드)
        style_keywords = [
            'fitted', 'loose', 'tight', 'relaxed', 'slim', 'wide', 'narrow',
            'long sleeves', 'short sleeves', 'sleeveless', 'strapless',
            'v-neck', 'round neck', 'crew neck', 'collar',
            'pockets', 'zipper', 'buttons', 'hook', 'elastic',
            'padded', 'underwired', 'moulded', 'support'
        ]
        
        found_styles = []
        for keyword in style_keywords:
            if keyword in caption_lower:
                # 원본 캡션에서 해당 부분 찾기
                pattern = r'[^.]*\b' + re.escape(keyword) + r'\b[^.]*'
                matches = re.findall(pattern, caption, re.IGNORECASE)
                if matches:
                    found_styles.append(matches[0].strip())
        
        if found_styles:
            style_fit = ', '.join(found_styles[:3])  # 최대 3개만
        
        # Other Details 추출 (나머지 중요한 정보)
        # Main Item, Materials, Style/Fit에 포함되지 않은 정보
        other_keywords = [
            'pocket', 'zipper', 'button', 'strap', 'trim', 'hem', 'waist',
            'denier', 'size', 'color', 'pattern', 'print', 'logo', 'brand'
        ]
        
        found_details = []
        for keyword in other_keywords:
            if keyword in caption_lower:
                pattern = r'[^.]*\b' + re.escape(keyword) + r'\b[^.]*'
                matches = re.findall(pattern, caption, re.IGNORECASE)
                for match in matches:
                    if match.strip() not in found_details and len(match.strip()) < 100:
                        found_details.append(match.strip())
        
        if found_details:
            other_details = ', '.join(found_details[:3])  # 최대 3개만
        
        # Main Item이 여전히 N/A이면 캡션의 첫 부분 사용
        if main_item == "N/A" and caption:
            # 첫 문장이나 첫 몇 단어 사용
            first_part = caption.split('.')[0].strip()
            if len(first_part) < 50:
                main_item = first_part
            else:
                main_item = ' '.join(caption.split()[:5])
        
        # 구조화된 형식으로 반환
        formatted = f"Brand: {brand}\nMain Item: {main_item}\nMaterials: {materials}\nCare: {care}\nStyle/Fit: {style_fit}\nOther Details: {other_details}"
        
        return formatted
    
    def _generate_simple_caption(self, features, image_path):
        """이미지 특징을 기반으로 간단한 설명 생성"""
        filename = Path(image_path).stem
        
        if features is None:
            return f"이미지 - {filename}"
        
        # 특징 벡터의 통계를 기반으로 기본 설명 생성
        try:
            mean_feature = features.mean()
            std_feature = features.std()
            
            # 간단한 휴리스틱 기반 설명
            if mean_feature > 0.5:
                base_desc = "밝고 선명한 이미지"
            else:
                base_desc = "이미지"
            
            return f"{base_desc} - {filename}"
        except:
            return f"이미지 - {filename}"


def process_images(image_folder_path, output_csv_path):
    """이미지 폴더의 모든 이미지를 처리하여 CSV 파일 생성"""
    
    # 지원하는 이미지 확장자
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    # 이미지 폴더 경로 확인
    image_folder = Path(image_folder_path)
    if not image_folder.exists():
        raise ValueError(f"이미지 폴더를 찾을 수 없습니다: {image_folder_path}")
    
    # 이미지 파일 찾기
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_folder.glob(f"*{ext}"))
        image_files.extend(image_folder.glob(f"*{ext.upper()}"))
    
    if not image_files:
        raise ValueError(f"이미지 폴더에 이미지 파일을 찾을 수 없습니다: {image_folder_path}")
    
    print(f"총 {len(image_files)}개의 이미지를 찾았습니다.")
    
    # 캡셔너 초기화
    captioner = SimpleImageCaptioner()
    
    # 결과 저장
    results = []
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] 처리 중: {image_path.name}")
        
        # product_id는 파일명 (확장자 제외)
        product_id = image_path.stem
        
        # 캡션 생성
        description = captioner.generate_caption(str(image_path))
        
        results.append({
            'product_id': product_id,
            'description': description
        })
    
    # CSV 파일로 저장
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['product_id', 'description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\n완료! 결과가 {output_csv_path}에 저장되었습니다.")
    print(f"총 {len(results)}개의 이미지가 처리되었습니다.")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='이미지 캡셔닝: 이미지 폴더의 이미지들에 대한 캡션을 생성합니다.')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='이미지 폴더의 경로')
    parser.add_argument('--output', '-o', type=str, default='image_captions.csv',
                       help='출력 CSV 파일 경로 (기본값: image_captions.csv)')
    
    args = parser.parse_args()
    
    try:
        process_images(args.input, args.output)
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())


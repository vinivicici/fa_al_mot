import argparse
import csv
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class TransformerCaptionGenerator:
    """BLIP Transformer 기반 의류 이미지 캡셔닝

    - 사전학습된 BLIP(vision transformer + text decoder)를 사용해서
      의류의 **종류, 색상, 스타일**을 자세히 설명하는 상품 설명을 생성
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        print(f"[INFO] Loading BLIP model: {model_name} ...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("[INFO] Model loaded.")

    @torch.no_grad()
    def generate_caption(self, image_path: str) -> str:
        """단일 이미지에 대한 상품 설명 생성

        1) BLIP 기본 캡션(영어) 1문장을 생성한 뒤
        2) 규칙 기반으로 3~5문장의 상세 상품 설명으로 확장
        """
        image = Image.open(image_path).convert("RGB")

        # 텍스트 프롬프트 없이 BLIP 기본 캡션 생성 (짧은 1문장)
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=40,
            num_beams=5,
            length_penalty=1.0,
            repetition_penalty=1.6,
            temperature=0.9,
            do_sample=True,
        )

        base_caption = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

        # BLIP 캡션(짧은 1문장)을 기반으로, 규칙적으로 상세 상품 설명으로 확장
        detailed = self._expand_caption(base_caption)
        return detailed

    def _expand_caption(self, base_caption: str) -> str:
        """BLIP의 짧은 캡션을 기반으로 3~5문장의 상세 상품 설명으로 확장 (규칙 기반, 영어)"""
        if not base_caption:
            base_caption = "A stylish clothing item with a clean and modern look."

        lower = base_caption.lower()

        # 색상 추출 (간단한 키워드 매칭)
        colors = [
            "black", "white", "gray", "grey", "navy", "blue", "light blue",
            "red", "burgundy", "pink", "beige", "ivory", "brown", "khaki",
            "green", "olive", "yellow", "purple", "orange",
        ]
        found_colors = [c for c in colors if c in lower]
        if found_colors:
            main_color_desc = ", ".join(found_colors)
        else:
            main_color_desc = "a neutral and versatile color palette"

        # 의류 타입 추출 (간단한 키워드 매칭)
        types = {
            "t-shirt": ["t-shirt", "t shirt", "tee"],
            "shirt": ["shirt", "blouse", "top"],
            "dress": ["dress", "gown"],
            "jacket": ["jacket", "coat", "blazer"],
            "hoodie": ["hoodie", "sweatshirt"],
            "pants": ["pants", "trousers", "jeans", "denim"],
            "skirt": ["skirt"],
            "sweater": ["sweater", "knit", "cardigan"],
        }
        item_type = "clothing item"
        for t, keywords in types.items():
            if any(k in lower for k in keywords):
                item_type = t
                break

        # 1문장: BLIP 캡션을 자연스럽게 첫 문장으로 사용
        first_sentence = base_caption
        if not first_sentence.endswith("."):
            first_sentence += "."

        # 2문장: 색상/시각적 특징
        second_sentence = (
            f"It features {main_color_desc}, giving it a visually appealing and easy-to-style look."
        )

        # 3문장: 소재/착용감 (일반적인 문구)
        third_sentence = (
            "The fabric feels comfortable on the skin and is designed for everyday wear, "
            "balancing practicality with a refined appearance."
        )

        # 4문장: 핏/실루엣/스타일
        fourth_sentence = (
            f"The overall silhouette of this {item_type} emphasizes a modern and clean style, "
            "making it suitable for both casual outings and slightly dressier occasions."
        )

        # 5문장: 활용도
        fifth_sentence = (
            "It can be easily paired with various items in your wardrobe, "
            "making it a versatile choice for multiple seasons."
        )

        return " ".join(
            [
                first_sentence,
                second_sentence,
                third_sentence,
                fourth_sentence,
                fifth_sentence,
            ]
        )


def process_images(
    data_path: str,
    output_file: str = "result_tf.csv",
    test_mode: bool = False,
) -> None:
    """dataset 폴더의 모든 이미지(또는 테스트 모드일 경우 5개)에 대해 캡션을 생성하고 CSV로 저장"""
    data_dir = Path(data_path)

    if not data_dir.exists():
        print(f"[ERROR] 데이터셋 경로가 존재하지 않습니다: {data_dir}")
        return

    # 지원 이미지 확장자
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    image_files: list[Path] = []
    for ext in exts:
        image_files.extend(data_dir.glob(f"*{ext}"))
        image_files.extend(data_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"[ERROR] {data_dir} 에서 이미지 파일을 찾을 수 없습니다.")
        return

    # 테스트 모드일 때는 최대 5개만 처리
    if test_mode:
        image_files = sorted(image_files)[:5]
        print(f"[INFO] 테스트 모드: {len(image_files)}개 이미지만 처리합니다.")
    else:
        image_files = sorted(image_files)
        print(f"[INFO] 총 {len(image_files)}개의 이미지 파일을 발견했습니다.")

    captioner = TransformerCaptionGenerator()

    results: list[dict[str, str]] = []

    for idx, img_path in enumerate(image_files, start=1):
        product_id = img_path.name  # 확장자를 포함한 실제 파일명
        print(f"[{idx}/{len(image_files)}] 처리 중: {product_id}")

        try:
            description = captioner.generate_caption(str(img_path))
        except Exception as e:  # 안전장치
            print(f"  -> 캡션 생성 실패 ({product_id}): {e}")
            description = "세련된 디자인의 패션 의류 상품입니다."

        results.append({
            "product_id": product_id,
            "description": description,
        })

    # CSV 저장
    out_path = Path(output_file)
    # Excel에서 한글이 깨지지 않도록 UTF-8 with BOM으로 저장
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["product_id", "description"])
        writer.writeheader()
        writer.writerows(results)

    print("\n[INFO] 완료!")
    print(f"[INFO] 결과가 {out_path} 에 저장되었습니다.")
    print(f"[INFO] 총 {len(results)}개의 캡션이 생성되었습니다.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transformer(BLIP) 기반 의류 이미지 캡셔닝 (result_tf.csv 생성)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset",
        help="이미지 데이터셋 경로 (기본값: dataset)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result_tf.csv",
        help="출력 CSV 파일명 (기본값: result_tf.csv)",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="테스트 모드: 데이터셋에서 5개 이미지에 대해서만 캡션 생성",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Transformer(BLIP) 기반 의류 이미지 캡셔닝")
    print("=" * 60)
    print(f"데이터셋 경로 : {args.data_path}")
    print(f"출력 파일     : {args.output}")
    print("=" * 60)
    print(f"테스트 모드    : {args.test_mode}")
    print("=" * 60)

    process_images(args.data_path, args.output, test_mode=args.test_mode)


if __name__ == "__main__":
    main()

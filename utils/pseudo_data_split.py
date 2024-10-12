import os
import json
import numpy as np
import random

# 파일 경로 설정
annotation = "/data/ephemeral/home/dataset/train_with_pseudo_labels.json"
train_path = '/data/ephemeral/home/dataset/pseudo_data_split/train_data.json'
val_path = '/data/ephemeral/home/dataset/pseudo_data_split/val_data.json'

# JSON 파일 읽기
with open(annotation) as f:
    data = json.load(f)

# 기존 데이터에서 pseudo label이 아닌 이미지들만 val set을 구성
non_pseudo_annotations = [ann for ann in data['annotations'] if not ann.get('is_pseudo', False)]
non_pseudo_image_ids = set(ann['image_id'] for ann in non_pseudo_annotations)

# 이미지 정보를 가져오고, 중복되지 않게 val set을 20%로 나눔
non_pseudo_images = [img for img in data['images'] if img['id'] in non_pseudo_image_ids]
random.shuffle(non_pseudo_images)

# 이미지 수 기준으로 80:20 비율로 나누기
split_idx = int(0.2 * len(non_pseudo_images))
val_images = non_pseudo_images[:split_idx]  # 20% validation images
train_images = non_pseudo_images[split_idx:]  # 나머지 80% train images

# Validation set 구성 (원래 데이터만 사용)
val_data = {
    "images": val_images,
    "categories": data["categories"],
    "annotations": [ann for ann in non_pseudo_annotations if ann['image_id'] in set(img['id'] for img in val_images)]
}

# Train set 구성 (원래 데이터 + pseudo label)
train_data = {
    "images": data['images'],  # 모든 이미지 사용
    "categories": data['categories'],
    "annotations": data['annotations']  # 모든 어노테이션 (pseudo 포함)
}

# 결과 저장
os.makedirs(os.path.dirname(train_path), exist_ok=True)
with open(train_path, "w") as f:
    json.dump(train_data, f, indent=4)

with open(val_path, "w") as f:
    json.dump(val_data, f, indent=4)

# 데이터의 개수 출력
print(f"Train set: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
print(f"Validation set: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")

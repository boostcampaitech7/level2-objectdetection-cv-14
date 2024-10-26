import os
import os
import json
import random
import argparse
# val set에 pseudo label이 들어가지 않도록 data split 

def split_data(annotation_file, train_path, val_path):
    # JSON 파일 읽기
    with open(annotation_file) as f:
        data = json.load(f)

    # 기존 데이터에서 pseudo label이 아닌 이미지들만 val set을 구성
    non_pseudo_annotations = [ann for ann in data['annotations'] if not ann.get('is_pseudo', False)]
    non_pseudo_image_ids = set(ann['image_id'] for ann in non_pseudo_annotations)
    
    # set을 list로 변환
    non_pseudo_image_ids_list = list(non_pseudo_image_ids)

    # 이미지 정보를 가져오고, 중복되지 않게 val set을 20%로 나눔
    non_pseudo_images = [img for img in data['images'] if img['id'] in non_pseudo_image_ids]
    random.shuffle(non_pseudo_images)

    # 이미지 수 기준으로 80:20 비율로 나누기
    split_idx = int(0.2 * len(non_pseudo_images))
    val_images = non_pseudo_images[:split_idx]  # 20% validation images
    val_image_ids = set(img['id'] for img in val_images)

    train_images = [img for img in data['images'] if img['id'] not in val_image_ids]  # validation set 이미지 제외

    # Validation set 구성 (원래 데이터만 사용)
    val_data = {
        "images": val_images,
        "categories": data["categories"],
        "annotations": [ann for ann in non_pseudo_annotations if ann['image_id'] in val_image_ids]
    }

    # Train set 구성 (validation set을 제외한 나머지 이미지 사용, pseudo 포함)
    train_data = {
        "images": train_images,  # validation set에 포함되지 않은 이미지들만 사용
        "categories": data['categories'],
        "annotations": [ann for ann in data['annotations'] if ann['image_id'] not in val_image_ids]  # validation set 이미지 제외
    }

    # 결과 저장
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=4)

    with open(val_path, "w") as f:
        json.dump(val_data, f, indent=4)

    # Train과 Val set의 이미지 ID가 겹치지 않는지 확인
    train_image_ids = set(img['id'] for img in train_data['images'])
    val_image_ids = set(img['id'] for img in val_data['images'])
    intersection = train_image_ids.intersection(val_image_ids)

    # 교집합이 비어 있으면 문제 없음, 교집합이 있으면 중복된 이미지 존재
    if len(intersection) == 0:
        print("Train set과 Validation set에 중복된 이미지가 없습니다.")
    else:
        print(f"Train set과 Validation set에 중복된 이미지가 있습니다: {len(intersection)}개")
        print(f"중복된 이미지 ID: {list(intersection)}")

    # 데이터의 개수 출력
    print(f"Train set: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
    print(f"Validation set: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")

if __name__ == "__main__":
    # Argument parser 설정
    parser = argparse.ArgumentParser(description="Split dataset into train and validation sets, excluding pseudo labels from the validation set.")
    parser.add_argument("annotation_file", type=str, help="Path to the input JSON annotation file")
    parser.add_argument("train_path", type=str, help="Path to save the train JSON file")
    parser.add_argument("val_path", type=str, help="Path to save the validation JSON file")

    args = parser.parse_args()

    # 파일 경로를 인자로 받아서 함수 호출
    split_data(args.annotation_file, args.train_path, args.val_path)
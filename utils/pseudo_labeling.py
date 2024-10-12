import json
import pandas as pd
import numpy as np

# 1. Train 데이터 로드
with open('/data/ephemeral/home/dataset/train.json', 'r') as f:
    train_data = json.load(f)

# 2. Test 데이터 예측 결과 CSV 파일 로드
pseudo_labels = pd.read_csv('./work_dirs/origin_RetinaNet/realorigin_retinanet.csv')  # Test 데이터 추론 결과

# Confidence score가 0.65 이상인 Pseudo Label 필터링 함수 정의
def filter_by_confidence(pred_string, threshold=0.6):
    try:
        pred_list = pred_string.split()
        # pred_list가 올바른 길이를 가지는지 확인하고 confidence score 기준으로 필터링
        return len(pred_list) > 1 and float(pred_list[1]) >= threshold
    except Exception as e:
        print(f"Error processing: {pred_string}, Error: {e}")
        return False

# 필터링을 적용
pseudo_labels = pseudo_labels[pseudo_labels['PredictionString'].apply(
    lambda x: isinstance(x, str) and filter_by_confidence(x, threshold=0.6))]

# 기존 train.json에서 마지막 이미지 및 어노테이션 ID 확인
last_image_id = max([img['id'] for img in train_data['images']])
last_annotation_id = max([ann['id'] for ann in train_data['annotations']])

# 새로운 이미지 및 어노테이션을 추가하기 위한 오프셋 설정
next_image_id = last_image_id + 1
next_annotation_id = last_annotation_id + 1

# COCO 형식의 새로운 어노테이션 정보를 저장할 리스트
new_images = []
new_annotations = []

# 필터링된 Pseudo Label을 COCO 형식으로 변환
for idx, row in pseudo_labels.iterrows():
    image_id = next_image_id + idx  # 새로운 image_id 부여
    image_name = row['image_id']  # Pseudo Label의 이미지 파일명
    prediction_string = row['PredictionString'].split()

    # Pseudo Label에서 class_id, Confidence, BBox, Class 정보를 파싱
    for i in range(0, len(prediction_string), 6):
        try:
            class_id = int(prediction_string[i])  # class_id는 첫 번째 값
            confidence = float(prediction_string[i + 1])  # 두 번째 값이 confidence

            x_min = float(prediction_string[i + 2])
            y_min = float(prediction_string[i + 3])
            x_max = float(prediction_string[i + 4])
            y_max = float(prediction_string[i + 5])

            # COCO 형식의 Bounding Box: [x_min, y_min, width, height]
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            area = (x_max - x_min) * (y_max - y_min)

            # 이미지 정보 추가
            new_images.append({
                "id": image_id,
                "file_name": image_name,
                "width": 1024,
                "height": 1024
            })

            # 어노테이션 정보 추가 (is_pseudo 필드 포함)
            new_annotations.append({
                "id": next_annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "is_pseudo": True,  # Pseudo Label임을 표시
                "score": confidence  # 추가적인 confidence 정보
            })
            next_annotation_id += 1
        except ValueError as e:
            print(f"Error in row {idx}: {e}, PredictionString: {prediction_string}")
            continue


# 기존 train.json에 새로운 이미지 및 어노테이션 병합
train_data['images'].extend(new_images)
train_data['annotations'].extend(new_annotations)

# 6. 병합된 데이터를 새로운 train_with_pseudo_labels.json으로 저장
with open('/data/ephemeral/home/dataset/train_with_pseudo_labels.json', 'w') as f:
    json.dump(train_data, f, indent=4)

print("Pseudo Label이 추가된 train_with_pseudo_labels.json 파일이 생성되었습니다.")

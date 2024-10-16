import json
import pandas as pd
import numpy as np
from collections import defaultdict

# IoU 계산 함수 정의
def calculate_iou(box1, box2):
    # box = [x_min, y_min, x_max, y_max]
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 교차 영역 계산
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # 각 바운딩 박스의 영역 계산
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # IoU 계산
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

# Pseudo Label CSV 파일 경로 설정
model_a_path ='/data/ephemeral/home/yj/level2-objectdetection-cv-14/mmdetection/swin_inference.csv'
model_b_path ='/data/ephemeral/home/yj/level2-objectdetection-cv-14/mmdetection/convnext.csv'

# 모델 결과 불러오기
model_a = pd.read_csv(model_a_path)
model_b = pd.read_csv(model_b_path)

# 모든 모델의 예측을 리스트로 저장
all_predictions = [model_a, model_b]

# 이미지별 바운딩 박스를 저장할 딕셔너리
bbox_dict = defaultdict(list)

# 각 모델의 예측을 bbox_dict에 저장
for predictions in all_predictions:
    for _, row in predictions.iterrows():
        image_id = row['image_id']
        if pd.isna(row['PredictionString']):
            continue
        bboxes = row['PredictionString'].split()
        for i in range(0, len(bboxes), 6):
            class_id = int(bboxes[i])
            confidence = float(bboxes[i + 1])
            x_min = float(bboxes[i + 2])
            y_min = float(bboxes[i + 3])
            x_max = float(bboxes[i + 4])
            y_max = float(bboxes[i + 5])
            bbox_dict[image_id].append([class_id, confidence, x_min, y_min, x_max, y_max])

# IoU 기반 매칭 및 Weighted Average 계산
pseudo_images = []
pseudo_annotations = []
next_image_id = 0
next_annotation_id = 0

for image_id, bboxes in bbox_dict.items():
    matched_boxes = []
    while bboxes:
        base_box = bboxes.pop(0)
        class_id, base_conf, x_min, y_min, x_max, y_max = base_box
        matched = [base_box]

        # 나머지 바운딩 박스와 IoU 비교
        for other_bbox in bboxes[:]:
            if other_bbox[0] == class_id:  # 동일한 클래스에 대해서만 매칭
                iou = calculate_iou(base_box[2:], other_bbox[2:])
                if iou > 0.5:  # IoU threshold
                    matched.append(other_bbox)
                    bboxes.remove(other_bbox)

        # 첫 번째 바운딩 박스의 좌표 사용, 가중 평균 신뢰도 계산 유지
        total_conf = sum([box[1] for box in matched])
        avg_conf = total_conf / len(matched)
        
        if avg_conf >= 0.6:  # 신뢰도 기준 필터링
            pseudo_annotations.append({
                "id": next_annotation_id,
                "image_id": next_image_id,
                "category_id": class_id,
                "bbox": [round(x_min, 1), round(y_min, 1), round(x_max - x_min, 1), round(y_max - y_min, 1)],
                "area": round((x_max - x_min) * (y_max - y_min), 2),
                "iscrowd": 0,
                "is_pseudo": True,
                "score": avg_conf
            })
            next_annotation_id += 1

    # 새로운 이미지 정보 추가
    pseudo_images.append({
        "id": next_image_id,
        "file_name": image_id,
        "width": 1024,
        "height": 1024,
        "license": 0,
        "flickr_url": None,
        "coco_url": None,
        "date_captured": "2020-12-12 15:19:33",
        "is_pseudo": True
    })
    next_image_id += 1

# 결과 JSON 파일로 저장
data = {
    "images": pseudo_images,
    "annotations": pseudo_annotations,
    "categories": [
        {"id": 0, "name": "General trash", "supercategory": "General trash"},
        {"id": 1, "name": "Paper", "supercategory": "Paper"},
        {"id": 2, "name": "Paper pack", "supercategory": "Paper pack"},
        {"id": 3, "name": "Metal", "supercategory": "Metal"},
        {"id": 4, "name": "Glass", "supercategory": "Glass"},
        {"id": 5, "name": "Plastic", "supercategory": "Plastic"},
        {"id": 6, "name": "Styrofoam", "supercategory": "Styrofoam"},
        {"id": 7, "name": "Plastic bag", "supercategory": "Plastic bag"},
        {"id": 8, "name": "Battery", "supercategory": "Battery"},
        {"id": 9, "name": "Clothing", "supercategory": "Clothing"}
    ]
}

output_path = '/data/ephemeral/home/dataset/pseudo_labels.json'
with open(output_path, 'w') as f:
    json.dump(data, f, indent=4)

print(f'Pseudo Label JSON 파일이 {output_path}에 저장되었습니다.')

# 체크할만한 정보 출력
print("\n===== 체크 정보 =====")
print(f"총 이미지 수: {len(pseudo_images)}")
print(f"총 어노테이션 수: {len(pseudo_annotations)}")
print(f"평균 바운딩 박스 수: {len(pseudo_annotations) / len(pseudo_images) if len(pseudo_images) > 0 else 0:.2f}")
print(f"최종 어노테이션 신뢰도 평균: {np.mean([ann['score'] for ann in pseudo_annotations]) if pseudo_annotations else 0:.2f}")
print(f"최종 어노테이션 면적 평균: {np.mean([ann['area'] for ann in pseudo_annotations]) if pseudo_annotations else 0:.2f}")

# 기존 train.json과 pseudo_labels.json을 합쳐서 새로운 JSON 파일 생성
train_json_path = '/data/ephemeral/home/dataset/train.json'
with open(train_json_path, 'r') as f:
    train_data = json.load(f)

# 기존 train 데이터의 마지막 ID 확인
last_image_id = max([img['id'] for img in train_data['images']]) if train_data['images'] else -1
last_annotation_id = max([ann['id'] for ann in train_data['annotations']]) if train_data['annotations'] else -1

# Pseudo Label의 ID를 기존 데이터의 마지막 ID 뒤에서 시작하도록 설정
next_image_id = last_image_id + 1
next_annotation_id = last_annotation_id + 1

# 기존 train 데이터에 pseudo label 데이터 병합
train_data['images'].extend(pseudo_images)
train_data['annotations'].extend(pseudo_annotations)

# 새로운 JSON 파일로 저장
combined_output_path = 'combined_train_pseudo_labels.json'
with open(combined_output_path, 'w') as f:
    json.dump(train_data, f, indent=4)

print(f'Combined JSON 파일이 {combined_output_path}에 저장되었습니다.')

# Combined JSON 파일의 이미지 및 어노테이션 개수 출력
print("===== Combined JSON 체크 정보 =====")
print(f"총 이미지 수: {len(train_data['images'])}")
print(f"총 어노테이션 수: {len(train_data['annotations'])}")
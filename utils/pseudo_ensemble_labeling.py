import json
import pandas as pd
import numpy as np
from collections import defaultdict
import random
import argparse 
# 2개의 모델 csv을 가져와 이를 앙상블하여 pseudo label을 생성하는 코드


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

# 데이터 로드 및 처리 함수
def process_pseudo_labels(model_a_path, model_b_path, train_json_path, output_path):
    # 모델 결과 불러오기
    model_a = pd.read_csv(model_a_path)
    model_b = pd.read_csv(model_b_path)
    all_predictions = [model_a, model_b]

    bbox_dict = defaultdict(list)
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

    with open(train_json_path, 'r') as f:
        train_data = json.load(f)

    last_image_id = max([img['id'] for img in train_data['images']]) if train_data['images'] else -1
    last_annotation_id = max([ann['id'] for ann in train_data['annotations']]) if train_data['annotations'] else -1
    next_annotation_id = last_annotation_id + 1

    pseudo_annotations = []
    for image_id, bboxes in bbox_dict.items():
        while bboxes:
            base_box = bboxes.pop(0)
            class_id, base_conf, x_min, y_min, x_max, y_max = base_box
            matched = [base_box]
            for other_bbox in bboxes[:]:
                if other_bbox[0] == class_id:
                    iou = calculate_iou(base_box[2:], other_bbox[2:])
                    if iou > 0.5:
                        matched.append(other_bbox)
                        bboxes.remove(other_bbox)

            total_conf = sum([box[1] for box in matched])
            avg_conf = total_conf / len(matched)
            
            if avg_conf >= 0.6:
                pseudo_annotations.append({
                    "image_id": image_id,
                    "category_id": class_id,
                    "area": round((x_max - x_min) * (y_max - y_min), 2),
                    "bbox": [round(x_min, 1), round(y_min, 1), round(x_max - x_min, 1), round(y_max - y_min, 1)],
                    "iscrowd": 0,
                    "id": next_annotation_id,
                    "score": avg_conf,
                    "is_pseudo": True
                })
                next_annotation_id += 1

    data = {
        "images": train_data['images'],
        "annotations": sorted(pseudo_annotations, key=lambda x: x['id']),
        "categories": train_data["categories"]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    total_images = len(train_data['images'])
    total_annotations = len(train_data['annotations'])
    combined_total_images = len(data['images'])
    average_bboxes = total_annotations / total_images if total_images > 0 else 0
    average_confidence = np.mean([ann['score'] for ann in pseudo_annotations]) if pseudo_annotations else 0
    average_area = np.mean([ann['area'] for ann in pseudo_annotations]) if pseudo_annotations else 0

    print(f'Pseudo Label JSON 파일이 {output_path}에 저장되었습니다.')
    print("===== 체크 정보 =====")
    print(f"총 이미지 수: {total_images}")
    print(f"총 어노테이션 수: {total_annotations}")
    print(f"평균 바운딩 박스 수: {average_bboxes:.2f}")
    print(f"최종 어노테이션 신뢰도 평균: {average_confidence:.2f}")
    print(f"최종 어노테이션 면적 평균: {average_area:.2f}")

    annotation_ids = [ann['id'] for ann in pseudo_annotations]
    if len(annotation_ids) == len(set(annotation_ids)):
        print("모든 어노테이션 ID가 고유합니다.")
    else:
        print("중복된 어노테이션 ID가 존재합니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pseudo labels and generate JSON output.")
    parser.add_argument("model_a_path", type=str, help="Path to model A CSV file")
    parser.add_argument("model_b_path", type=str, help="Path to model B CSV file")
    parser.add_argument("train_json_path", type=str, help="Path to the train JSON file")
    parser.add_argument("output_path", type=str, help="Path to save the output JSON file")

    args = parser.parse_args()

    process_pseudo_labels(args.model_a_path, args.model_b_path, args.train_json_path, args.output_path)
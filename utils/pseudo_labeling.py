import json
import pandas as pd
import numpy as np
import argparse
# pseudo label을 생성새 train.json과 합치는 코드 

# Confidence score가 일정 수준 이상인 Pseudo Label 필터링 함수 정의
def filter_by_confidence(pred_string, threshold=0.65):
    try:
        pred_list = pred_string.split()
        return len(pred_list) > 1 and float(pred_list[1]) >= threshold
    except Exception as e:
        print(f"Error processing: {pred_string}, Error: {e}")
        return False

# Pseudo Label 데이터를 COCO 형식으로 변환 및 병합하는 함수 정의
def add_pseudo_labels(train_json_path, pseudo_labels_path, output_path, confidence_threshold=0.65):
    # 1. Train 데이터 로드
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)

    # 2. Test 데이터 예측 결과 CSV 파일 로드
    pseudo_labels = pd.read_csv(pseudo_labels_path)

    # Confidence score가 threshold 이상인 Pseudo Label 필터링
    pseudo_labels = pseudo_labels[pseudo_labels['PredictionString'].apply(
        lambda x: isinstance(x, str) and filter_by_confidence(x, threshold=confidence_threshold))]

    # 기존 train.json에서 마지막 이미지 및 어노테이션 ID 확인
    last_image_id = max([img['id'] for img in train_data['images']])
    last_annotation_id = max([ann['id'] for ann in train_data['annotations']])

    # 새로운 이미지 및 어노테이션을 추가하기 위한 오프셋 설정
    next_image_id = last_image_id + 1
    next_annotation_id = last_annotation_id + 1

    # COCO 형식의 새로운 어노테이션 정보를 저장할 리스트
    new_images = {}
    new_annotations = []

    # 필터링된 Pseudo Label을 COCO 형식으로 변환
    for idx, row in pseudo_labels.iterrows():
        image_name = row['image_id']
        prediction_string = row['PredictionString'].split()
        
        # 기존에 이미지가 추가된 적이 있는지 확인
        if image_name not in new_images:
            new_images[image_name] = {
                "id": next_image_id,
                "file_name": image_name,
                "width": 1024,
                "height": 1024
            }
            image_id = next_image_id
            next_image_id += 1
        else:
            image_id = new_images[image_name]['id']

        # Pseudo Label에서 class_id, confidence, BBox 정보 파싱
        for i in range(0, len(prediction_string), 6):
            try:
                class_id = int(prediction_string[i])
                confidence = float(prediction_string[i + 1])
                x_min = float(prediction_string[i + 2])
                y_min = float(prediction_string[i + 3])
                x_max = float(prediction_string[i + 4])
                y_max = float(prediction_string[i + 5])

                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = (x_max - x_min) * (y_max - y_min)

                new_annotations.append({
                    "id": next_annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "is_pseudo": True,
                    "score": confidence
                })
                next_annotation_id += 1
            except ValueError as e:
                print(f"Error in row {idx}: {e}, PredictionString: {prediction_string}")
                continue

    # 추가된 이미지 및 어노테이션 병합
    train_data['images'].extend(new_images.values())
    train_data['annotations'].extend(new_annotations)

    # 병합된 데이터를 새로운 JSON 파일로 저장
    with open(output_path, 'w') as f:
        json.dump(train_data, f, indent=4)

    # 추가된 정보 출력
    print(f"Pseudo Label이 추가된 JSON 파일이 {output_path}에 저장되었습니다.")
    print(f"추가된 이미지 개수: {len(new_images)}")
    print(f"추가된 어노테이션 개수: {len(new_annotations)}")

if __name__ == "__main__":
    # Argument parser 설정
    parser = argparse.ArgumentParser(description="Add pseudo labels to train JSON.")
    parser.add_argument("train_json_path", type=str, help="Path to the train JSON file")
    parser.add_argument("pseudo_labels_path", type=str, help="Path to the pseudo labels CSV file")
    parser.add_argument("output_path", type=str, help="Path to save the output JSON file with added pseudo labels")
    parser.add_argument("--confidence_threshold", type=float, default=0.65, help="Confidence threshold for filtering pseudo labels")

    args = parser.parse_args()

    # 함수 호출
    add_pseudo_labels(args.train_json_path, args.pseudo_labels_path, args.output_path, args.confidence_threshold)


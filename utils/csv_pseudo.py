import json
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Pseudo label generation from ensemble results.')
parser.add_argument('--csv_path', type=str, required=True, help='Path to the ensemble result CSV file.')

args = parser.parse_args()

# CSV 파일 경로 설정 (명령줄에서 받은 인자 사용)
ensemble_csv_path = args.csv_path

# 앙상블 모델 결과 불러오기
ensemble_df = pd.read_csv(ensemble_csv_path)

# 이미지별 바운딩 박스를 저장할 딕셔너리
filtered_bbox_dict = {}
j =0
# 앙상블 결과에서 신뢰도가 0.6 이상인 바운딩 박스만 필터링
for _, row in ensemble_df.iterrows():
    image_id = row['image_id']
    if pd.isna(row['PredictionString']):
        continue
    bboxes = []
    split_box = row["PredictionString"].split()
    for k in range(0, len(split_box), 6) :
        box = [split_box[k],split_box[k+1],split_box[k+2],split_box[k+3],split_box[k+4],split_box[k+5]]
        bboxes.append(box)
    filtered_bboxes = []
    for i in range(len(bboxes)):
        j+=1
        class_id = int(bboxes[i][0])
        confidence = float(bboxes[i][1])
        x_min = float(bboxes[i][2])
        y_min = float(bboxes[i][3])
        x_max = float(bboxes[i][4])
        y_max = float(bboxes[i][5])
        if confidence >= 0.6:
            filtered_bboxes.append([class_id, confidence, x_min, y_min, x_max, y_max])
    if filtered_bboxes:
        filtered_bbox_dict[image_id] = filtered_bboxes

# 기존 train.json을 불러와서 ID 업데이트
train_json_path = '/data/ephemeral/home/dataset/train.json'
with open(train_json_path, 'r') as f:
    train_data = json.load(f)

# 기존 train 데이터의 마지막 ID 확인
last_image_id = max([img['id'] for img in train_data['images']]) if train_data['images'] else -1
last_annotation_id = max([ann['id'] for ann in train_data['annotations']]) if train_data['annotations'] else -1

# Pseudo Label의 ID를 기존 데이터의 마지막 ID 뒤에서 시작하도록 설정
next_annotation_id = last_annotation_id + 1

# 필터링된 바운딩 박스를 이용해 어노테이션 생성
pseudo_annotations = []
for image_id, bboxes in filtered_bbox_dict.items():
    for bbox in bboxes:
        class_id, confidence, x_min, y_min, x_max, y_max = bbox
        pseudo_annotations.append({
            "image_id": image_id,  # 기존 이미지 ID 사용
            "category_id": class_id,
            "area": round((x_max - x_min) * (y_max - y_min), 2),
            "bbox": [round(x_min, 1), round(y_min, 1), round(x_max - x_min, 1), round(y_max - y_min, 1)],
            "iscrowd": 0,
            "id": next_annotation_id,
            "score": confidence,
            "is_pseudo": True
        })
        next_annotation_id += 1

# 결과 JSON 파일로 저장
data = {
    "images": train_data['images'],
    "annotations": sorted(pseudo_annotations, key=lambda x: x['id']),  # ID 순서 정렬
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

output_path = '/data/ephemeral/home/dataset/csv_pseudo.json'
with open(output_path, 'w') as f:
    json.dump(data, f, indent=4)

# 출력 포맷

total_images = len(train_data['images'])
total_annotations = len(pseudo_annotations)
combined_total_images = len(data['images'])
average_bboxes = total_annotations / total_images if total_images > 0 else 0
average_confidence = np.mean([ann['score'] for ann in pseudo_annotations]) if pseudo_annotations else 0
average_area = np.mean([ann['area'] for ann in pseudo_annotations]) if pseudo_annotations else 0

print(f'Pseudo Label JSON 파일이 {output_path}에 저장되었습니다.')
print("===== 체크 정보 =====")
print(f"기존 어노테이션 수: {j}")
print(f"총 이미지 수: {total_images}")
print(f"총 어노테이션 수: {total_annotations}")
print(f"평균 바운딩 박스 수: {average_bboxes:.2f}")
print(f"최종 어노테이션 신뢰도 평균: {average_confidence:.2f}")
print(f"최종 어노테이션 면적 평균: {average_area:.2f}")

# 어노테이션 ID의 고유성 확인
annotation_ids = [ann['id'] for ann in pseudo_annotations]
if len(annotation_ids) == len(set(annotation_ids)):
    print("모든 어노테이션 ID가 고유합니다.")
else:
    print("중복된 어노테이션 ID가 존재합니다.")




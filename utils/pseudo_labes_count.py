import json
import pandas as pd


# 1. JSON 파일 불러오기
file_path = '/data/ephemeral/home/dataset/train_with_pseudo_labels.json'
with open(file_path, 'r') as f:
    train_data = json.load(f)

# 2. 어노테이션에서 is_pseudo가 True인 항목만 필터링하여 개수 세기
pseudo_labels_count = sum(1 for ann in train_data['annotations'] if 'is_pseudo' in ann and ann['is_pseudo'])

# 3. 결과 출력
print(f"Pseudo Label로 추가된 어노테이션 개수: {pseudo_labels_count}")

# 추론 결과 CSV 파일 로드 
pseudo_labels = pd.read_csv('./work_dirs/origin_RetinaNet/realorigin_retinanet.csv')

# 각 row에서 PredictionString을 분석하여 총 예측된 bounding box 개수 계산
total_annotations = 0

for idx, row in pseudo_labels.iterrows():
    prediction_string = row['PredictionString']
    
    # PredictionString이 문자열인지 확인하고 NaN 값 무시
    if isinstance(prediction_string, str):
        prediction_string = prediction_string.split()
        # 하나의 PredictionString에서 6개의 값이 한 번에 하나의 객체를 의미하므로 6으로 나누어 개수 계산
        total_annotations += len(prediction_string) // 6

print(f"Test 데이터에서 예측된 총 어노테이션 개수: {total_annotations}")



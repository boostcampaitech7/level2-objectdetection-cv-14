import json
import pandas as pd
import argparse 
# 생성된 pseudo label이 몇개인지 알아보는 코드 


def count_pseudo_labels(json_path):
    # JSON 파일 불러오기
    with open(json_path, 'r') as f:
        train_data = json.load(f)

    # 어노테이션에서 is_pseudo가 True인 항목만 필터링하여 개수 세기
    pseudo_labels_count = sum(1 for ann in train_data['annotations'] if 'is_pseudo' in ann and ann['is_pseudo'])
    return pseudo_labels_count

def count_total_annotations(csv_path):
    # CSV 파일 불러오기
    pseudo_labels = pd.read_csv(csv_path)

    # 각 row에서 PredictionString을 분석하여 총 예측된 bounding box 개수 계산
    total_annotations = 0
    for _, row in pseudo_labels.iterrows():
        prediction_string = row['PredictionString']
        
        # PredictionString이 문자열인지 확인하고 NaN 값 무시
        if isinstance(prediction_string, str):
            prediction_string = prediction_string.split()
            # 6개의 값이 하나의 객체를 의미하므로 6으로 나누어 개수 계산
            total_annotations += len(prediction_string) // 6

    return total_annotations

if __name__ == "__main__":
    # Argument parser 설정
    parser = argparse.ArgumentParser(description="Count pseudo labels and bounding boxes in JSON and CSV files.")
    parser.add_argument("json_path", type=str, help="Path to the JSON file containing pseudo labels")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing prediction results")

    args = parser.parse_args()

    # 함수 호출
    pseudo_labels_count = count_pseudo_labels(args.json_path)
    total_annotations = count_total_annotations(args.csv_path)

    # 결과 출력
    print(f"Pseudo Label로 추가된 어노테이션 개수: {pseudo_labels_count}")
    print(f"Test 데이터에서 예측된 총 어노테이션 개수: {total_annotations}")
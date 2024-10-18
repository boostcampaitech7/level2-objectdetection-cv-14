from ultralytics import YOLO
import csv
from pathlib import Path
            
def inference(model_path, output_file):
    model = YOLO(model_path)

    # 이미지 디렉토리
    image_dir = Path('./datasets/origin_dataset/test')

    with open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['PredictionString', 'image_id'])

        # 디렉토리 내의 이미지 파일을 정렬하여 순서대로 처리
        for image_path in sorted(image_dir.glob('*.jpg')):
            # 이미지에서 객체 탐지
            results = model(str(image_path))

            # 결과 처리 및 CSV 파일에 저장
            for r in results:
                result_str = ''
                for box, conf, cls_id in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                    x1, y1, x2, y2 = box
                    result_str += f"{int(cls_id)} {conf} {x1} {y1} {x2} {y2} "

                # 이미지 경로 재구성
                image_id = 'test/' + image_path.name
                # CSV 파일에 결과 작성
                csv_writer.writerow([result_str, image_id])
                
for i in range(1, 6):
    inference(f"./runs/detect/yolov8-fold[{i}]_2/weights/best.pt", f"sumbission_fold[{i}]")
from ultralytics import YOLO
import csv

model = YOLO("./runs/detect/train3/weights/best.pt")

for i in range(4871):
    image_path = f"./datasets/origin_dataset/test/{i:04d}.jpg"
    results = model.predict(
        image_path,
        save=False,
        imgsz=1024,
        conf=0.5,
        device="cuda",
    )
    content = ''
    for idx, j in enumerate(results[0].boxes) :
        pred_class = int(j.cls.item())
        conf = j.conf.item()
        x1, y1, x2, y2 = map(float, j.xyxy[0].tolist())
        content += f"{pred_class} {conf:.8f} {x1:.5f} {y1:.5f} {x2:.5f} {y2:.5f} "
        # print(content)
    
    with open('submission.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if i == 0:
            writer.writerow(['PredictionString', 'image_id'])
        writer.writerow([str(content), str(f"test/{i:04d}.jpg")])
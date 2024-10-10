import json
import os

def convert_coco_to_yolo(coco_json_path, output_dir):
    # COCO JSON 파일 로드
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
        print(len(coco_data['annotations']))
        
        id = 0
        content = ''
        for idx, item in enumerate(coco_data['annotations']):
            if item['image_id'] != id:
                filename = f"{id:04d}.txt"
                output_path = os.path.join(output_dir, filename)
                with open(output_path, 'w') as ff:
                    print(output_path)
                    ff.write(content)
                id = item['image_id']
                content = ''                    
                    
            category_id = str(item['category_id'])
            x_center = "{:.6f}".format((item['bbox'][0]*2 + item['bbox'][2])/2.0)
            y_center = "{:.6f}".format((item['bbox'][1]*2 + item['bbox'][3])/2.0)
            width = str(item['bbox'][1])
            height = str(item['bbox'][3])
            data = category_id + " " + x_center + " " + y_center + " " + width + " " + height + "\n"
            content += data
        
        filename = f"{id:04d}.txt"
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as ff:
            print(output_path)
            ff.write(content)

# 사용 예시
coco_json_path = './datasets/origin_dataset/train.json'
output_dir = './datasets/yolo_form_dataset/labels/train'
os.makedirs(output_dir, exist_ok=True)
convert_coco_to_yolo(coco_json_path, output_dir)

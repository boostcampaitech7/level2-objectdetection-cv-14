import json
import os
import pandas as pd

# confidence Threshold 설정
CONF_THR = 0.4
CONF_MEAN = 0.7


train_json_path = f'/data/ephemeral/home/dataset/train.json'

# test.json inference한 결과 경로 설정
submission_path = '/data/ephemeral/home/yj/level2-objectdetection-cv-14/mmdetection/work_dirs/origin_RetinaNet/realorigin_retinanet.csv'

# 결과물이 저장될 root와 이름 설정
output_root = '/data/ephemeral/home/dataset'
output_json_name = 'train_pseudo_mean_labels'
output_path = os.path.join(output_root, output_json_name + '.json')

# 기존 train.json 불러오기
with open(train_json_path) as f:
    train_json = json.load(f)
    
# submission.csv 파일 읽기 (pseudo label 데이터)
df = pd.read_csv(submission_path)
submission_list = df[['PredictionString', 'image_id']].values.tolist()


images = []
annotations = []

# train.json에서 마지막 image 및 annotation ID 가져오기
start_id = train_json['images'][-1]['id']
start_anno_id = train_json['annotations'][-1]['id']

print('pseudo labeling start...')

# submission_list에서 bbox와 image_name 정보 추출
for bboxes, image_name in submission_list:
    if not isinstance(bboxes, str):
        continue  # bboxes가 문자열이 아닌 경우 스킵
    # 새로운 image ID 생성
    start_id += 1
    bboxes_splited = bboxes.split()
    num_bbox = 0

    # 이미지 당 10개 box 이상이거나 평균 confidence가 낮으면 제거
    conf_box = 0
    conf_total = 0
    for i in range(1, len(bboxes_splited) + 1, 6):
        if float(bboxes_splited[i]) >= CONF_THR:
            conf_box += 1
            conf_total += float(bboxes_splited[i])

    if conf_box == 0:
        continue
    conf_mean = conf_total / conf_box

    if conf_mean < CONF_MEAN:
        continue

    # 새로운 annotation 생성
    for i in range(6, len(bboxes_splited) + 1, 6):
        bbox = bboxes_splited[i - 6:i]
        _class, conf, left, top, right, bottom = int(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4]), float(bbox[5])
        width, height = right - left, bottom - top
        area = round(width * height, 2)

        if conf < CONF_THR:
            continue

        start_anno_id += 1
        num_bbox += 1

        # 새로운 어노테이션 추가 (pseudo label로 구분)
        annotation = {
            'image_id': start_id,
            'category_id': _class,
            'area': area,
            'bbox': [round(left, 1), round(top, 1), round(width, 1), round(height, 1)],
            'iscrowd': 0,
            'id': start_anno_id,
            'is_pseudo': True  # pseudo label로 구분
        }
        annotations.append(annotation)

    if num_bbox != 0:
        # 새로운 이미지 추가 (pseudo label로 구분)
        image = {
            'width': 1024,
            'height': 1024,
            'file_name': image_name,
            'license': 0,
            'flickr_url': None,
            'coco_url': None,
            'date_captured': "2020-12-12 15:19:33",
            'id': start_id,
            'is_pseudo': True  # pseudo label로 구분
        }
        images.append(image)

print('pseudo labeling finished...\n')
print(f'{len(images)}개의 images가 추가되었습니다.')
print(f'{len(annotations)}개의 annotations이 추가되었습니다.\n')

# 기존 train.json에 pseudo label 데이터 추가
train_json['images'] += images
train_json['annotations'] += annotations

# output json저장
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(train_json, f, indent=4)

print(f'{output_path} 에 저장되었습니다.')
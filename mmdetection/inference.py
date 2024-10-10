# import mmcv
# from mmcv import Config
# from mmdet.datasets import (build_dataloader, build_dataset,
#                             replace_ImageToTensor)
# from mmdet.models import build_detector
# from mmdet.apis import single_gpu_test
# from mmcv.runner import load_checkpoint
# import os
# from mmcv.parallel import MMDataParallel
# import pandas as pd
# from pandas import DataFrame
# from pycocotools.coco import COCO
# import numpy as np
# import argparse

# # config file 들고오기
# parser = argparse.ArgumentParser()

# # config 파일 경로
# parser.add_argument('config', help='test config file path')

# # 결과 저장 경로
# parser.add_argument('-o', '--output', help='output file path')

# # 인자 받기
# args = parser.parse_args()
# if args.output is None:
#     args.output = os.path.join("./work_dirs", os.path.split(args.config)[1][:-3])

# classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
#            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# # config file 들고오기
# cfg = Config.fromfile(args.config)

# epoch = 'latest'

# # dataset config 수정
# cfg.data.test.classes = classes
# cfg.data.test.test_mode = True

# cfg.data.samples_per_gpu = 4

# cfg.seed=2024
# cfg.gpu_ids = [1]
# cfg.work_dir = args.output

# cfg.model.train_cfg = None

# # build dataset & dataloader
# dataset = build_dataset(cfg.data.test)
# data_loader = build_dataloader(
#         dataset,
#         samples_per_gpu=1,
#         workers_per_gpu=cfg.data.workers_per_gpu,
#         dist=False,
#         shuffle=False)

# # checkpoint path
# checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

# model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
# checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

# model.CLASSES = dataset.CLASSES
# model = MMDataParallel(model.cuda(), device_ids=[0])

# output = single_gpu_test(model, data_loader, show_score_thr=0.05) # output 계산

# # submission 양식에 맞게 output 후처리
# prediction_strings = []
# file_names = []
# coco = COCO(cfg.data.test.ann_file)
# img_ids = coco.getImgIds()

# class_num = 10
# for i, out in enumerate(output):
#     prediction_string = ''
#     image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
#     for j in range(class_num):
#         for o in out[j]:
#             prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
#                 o[2]) + ' ' + str(o[3]) + ' '
        
#     prediction_strings.append(prediction_string)
#     file_names.append(image_info['file_name'])


# submission = pd.DataFrame()
# submission['PredictionString'] = prediction_strings
# submission['image_id'] = file_names
# submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}.csv'), index=None)
# submission.head()

import pickle

# results.pkl 파일 경로를 지정하세요.
pkl_file = '/data/ephemeral/home/yj/level2-objectdetection-cv-14/mmdetection/results.pkl'

# 파일을 열고 내용을 불러옵니다.
with open(pkl_file, 'rb') as f:
    results = pickle.load(f)

# 결과의 일부를 출력하여 확인해 보세요.
print(results[:5])


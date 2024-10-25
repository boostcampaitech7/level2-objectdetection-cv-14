import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.Gsheet import Gsheet_param

# 모듈 import
from mmengine.config import Config
# from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device
import argparse


# config file 들고오기
parser = argparse.ArgumentParser()

# config 파일 경로
parser.add_argument('config', help='train config file path')

# 결과 저장 경로
parser.add_argument('-o', '--output', help='output file path')

# validation 여부 인자 
# -v or --validation이 입력되면 false를 출력한다.
parser.add_argument('-v', '--validation', help='Validation Check', action='store_false')

# 인자 받기
args = parser.parse_args()

if args.output is None:
    args.output = os.path.join("./work_dirs", os.path.split(args.config)[1][:-3])

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

cfg = Config.fromfile(args.config)

# dataset config 수정
cfg.data.train.classes = classes
cfg.data.val.classes = classes
cfg.data.test.classes = classes

cfg.seed = 2024
cfg.gpu_ids = [0]
cfg.work_dir = args.output
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
cfg.device = get_device()

# build_dataset
datasets = [build_dataset(cfg.data.train)]

# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
model.init_weights()

train_detector(model, datasets[0], cfg, distributed=False, validate=args.validation)

Gsheet_param(cfg, args.output)

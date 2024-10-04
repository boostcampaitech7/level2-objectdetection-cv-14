_base_ = "../../../configs/detr/detr_r50_8x2_150e_coco.py"

# data 관련 부분 수정
data_root = "../../../../dataset"

data = dict(
    train=dict(
        ann_file=data_root + 'train.json',
        img_prefix=data_root),
    test=dict(
        ann_file=data_root + 'test.json',
        img_prefix=data_root))


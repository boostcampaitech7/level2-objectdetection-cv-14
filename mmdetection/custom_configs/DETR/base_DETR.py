_base_ = "../../configs/detr/detr_r50_8x2_150e_coco.py"

# data 관련 부분 수정
data_root = "../../../dataset"

data = dict(
    train=dict(
        ann_file=data_root + 'train.json',
        img_prefix=data_root),
    test=dict(
        ann_file=data_root + 'test.json',
        img_prefix=data_root))


# model 관련 class 개수 수정
model = dict(
    bbox_head=dict(
        num_classes=10
    ),
    init_cfg=dict(
        type='Pretrained', 
        checkpoint='https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'
    )
)

checkpoint_config = dict(interval=-1)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[5])
runner = dict(type='EpochBasedRunner', max_epochs=20)


find_unused_parameters = True
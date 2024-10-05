_base_ = "../../configs/detr/detr_r50_8x2_150e_coco.py"

# data 관련 부분 수정
data_root = "../../../dataset"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type="Resize", img_scale=(512, 512), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    train=dict(
        ann_file=data_root + 'train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    test=dict(
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))


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
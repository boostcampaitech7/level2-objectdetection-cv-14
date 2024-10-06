_base_ = "../../configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py"

# data 관련 부분 수정
data_root = "../../../dataset"

data = dict(
    train=dict(
        ann_file=data_root + 'train.json',
        img_prefix=data_root),
    test=dict(
        ann_file=data_root + 'test.json',
        img_prefix=data_root))

model = dict(
    bbox_head=dict(
        num_classes=10
    ),
    init_cfg=dict(
        type="Pretrained",
        checkpoint="https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth"
    )
)


optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=1000,
                 warmup_ratio=0.001,
                 step=5,
                 gamma=0.5)
runner = dict(type='EpochBasedRunner', max_epochs=30)
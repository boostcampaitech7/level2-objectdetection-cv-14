_base_ = "../../configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py"

# data 관련 부분 수정
data_root = "../../../dataset/"

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

data = dict(
    train=dict(
        ann_file=data_root + 'sgkf/train_1fold.json',
        img_prefix=data_root,
        classes = classes),
    val=dict(
        ann_file=data_root + 'sgkf/val_1fold.json',
        img_prefix=data_root,
        classes = classes),
    test=dict(
        ann_file=data_root + 'sgkf/val_1fold.json',
        img_prefix=data_root,
        classes = classes))

evaluation = dict(interval=1, metric='bbox', save_best='auto')

model = dict(
    bbox_head=dict(
        num_classes=10,
        # box refine 기능 추가
        # 모든 디코더 레이어에서 바운딩 박스를 정제하는 기능
        with_box_refine=True,
        as_two_stage=True
    ),
    # init_cfg=dict(
    #     type="Pretrained",
        # checkpoint="https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth"
        
        # box refinement feature added model
        # checkpoint="https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.pth"

        # Basic Deformable DETR
        # checkpoint="https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth"
    # )
)

load_from = "https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth"


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
runner = dict(type='EpochBasedRunner', max_epochs=20)
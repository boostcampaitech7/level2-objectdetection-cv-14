_base_ = "./codino_models.py"

# dataset 관련
image_size = (1024, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict( # LSJ
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline))


# optimizer 관련
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

# learning policy
max_epochs = 15
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0,
        begin=0,
        T_max=max_epochs,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True)
]


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project' : 'Recycle-Object-Detection',
            'group' : 'CO-DINO-5scale-Swin',
            'name' : 'CO-DINO-5scale-Swin-1fold',
            'entity' : 'cv14_',
            'tags' : ['Swin384', '15Epoch', '5scale', 'LSJ']
         })]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_lsj_swin_large_1x_coco-3af73af2.pth'
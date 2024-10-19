# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    'models/Cascade_rcnn_ConvNeXt.py',
    'datasets/coco_detection_aug_3_fold5.py',
    'schedules/schedule_CosineAnnealing.py',
    'runtime.py'
]
multi_scale = [(w,w) for w in range(512, 1024+1, 32)]

albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(type='Flip',p=1.0),
            dict(type='RandomRotate90',p=1.0)
        ],
        p = 0.1
    ),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(type='RandomResizedCrop',height=1024, width=1024, scale=(0.5, 1.0), p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='ChannelShuffle',
                p=1.0),
            dict(
                type='RandomGamma',
                p=1.0),
            dict(
                type='RGBShift',
                p=1.0)
        ],
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
    ]

model = dict(
    backbone=dict(
        in_chans=3,
        depths=[3, 3, 27, 3], 
        dims=[256, 512, 1024, 2048], 
        drop_path_rate=0.8,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    neck=dict(in_channels=[256, 512, 1024, 2048]),
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize', 
        img_scale=multi_scale,
        multiscale_mode='value',
        keep_ratio=True,
        ),
    dict(
    type='Albu',
    transforms=albu_train_transforms,
    bbox_params=dict(
        type='BboxParams',
        format='pascal_voc',
        label_fields=['gt_labels'],
        min_visibility=0.0,
        filter_lost_elements=True),
    keymap={
        'img': 'image',
        'gt_bboxes': 'bboxes'
    },
    update_pad_shape=False,
    skip_img_without_anno=True
    ),
    # dict(type='AutoAugment',
    #      policies=[
    #          [
    #              dict(type='Resize',
    #                   img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
    #                              (608, 1333), (640, 1333), (672, 1333), (704, 1333),
    #                              (736, 1333), (768, 1333), (800, 1333)],
    #                   multiscale_mode='value',
    #                   keep_ratio=True)
    #          ],
    #          [
    #              dict(type='Resize',
    #                   img_scale=[(400, 1333), (500, 1333), (600, 1333)],
    #                   multiscale_mode='value',
    #                   keep_ratio=True),
    #              dict(type='RandomCrop',
    #                   crop_type='absolute_range',
    #                   crop_size=(384, 600),
    #                   allow_negative_crop=True),
    #              dict(type='Resize',
    #                   img_scale=[(480, 1333), (512, 1333), (544, 1333),
    #                              (576, 1333), (608, 1333), (640, 1333),
    #                              (672, 1333), (704, 1333), (736, 1333),
    #                              (768, 1333), (800, 1333)],
    #                   multiscale_mode='value',
    #                   override=True,
    #                   keep_ratio=True)
    #          ]
    #      ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(pipeline=train_pipeline), samples_per_gpu=2)

optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW', 
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.7,
                                'decay_type': 'layer_wise',
                                'num_layers': 12})
#lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=12)

# do not use mmdet version fp16
fp16 = None
load_from='https://dl.fbaipublicfiles.com/convnext/coco/cascade_mask_rcnn_convnext_xlarge_22k_3x.pth'
# please install mmcls>=0.22.0
# import mmcls.models to trigger register_module in mmcls
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa
checkpoint_ConvNext_Base = 'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth'
checkpoint_ConvNext_B_22k = 'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth'

checkpoint_ConvNext_Large = 'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth'
checkpoint_ConvNext_L_22k = 'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth'

checkpoint_ConvNext_XL = 'https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth'
checkpoint_ConvNext_XL_22k = 'https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth'


model = dict(
    type='ATSS',
    backbone=dict(
        backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768], 
        drop_path_rate=0.2,
        layer_scale_init_value=1e-6,
        out_indices=[0, 1, 2, 3],
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_ConvNext_Base))),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
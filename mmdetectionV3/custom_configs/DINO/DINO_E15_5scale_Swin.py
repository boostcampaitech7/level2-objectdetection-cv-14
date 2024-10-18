_base_ = "./DINO_E12_5scale_Swin.py"

# learning policy
max_epochs = 15
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-5,  # 미세 조정을 위한 낮은 학습률
        weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.05)  # 백본의 학습률을 낮춰서 미세 조정
    })
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[10, 13],  # 10 에폭과 13 에폭 후에 점진적으로 감소
        gamma=0.1)
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
             'project' : 'Recycle-Object-Detection',
             'group' : 'DINO-5scale-Swin',
             'name' : 'Base-DINO-5Scale-Swin-Improve',
             'entity' : 'cv14_',
             'tags' : ['Swin384', '15Epoch', '5scale']
         })
]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')


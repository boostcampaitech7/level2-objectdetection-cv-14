_base_ = "./CO_DINO_E15_5scale_Swin_lsj.py"

train_dataloader = dict(
    dataset=dict(
        ann_file='sgkf_5_14/train_3fold.json'
    )
)

val_dataloader = dict(
    dataset=dict(
        ann_file='sgkf_5_14/val_3fold.json'
    )
)

val_evaluator = dict(
    ann_file=_base_.data_root + 'sgkf_5_14/val_3fold.json',
)

# learning policy
max_epochs = 15
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11, 13],
        gamma=0.1)
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project' : 'Recycle-Object-Detection',
            'group' : 'CO-DINO-5scale-Swin',
            'name' : 'CO-DINO-5scale-Swin-3fold-StepScheduler',
            'entity' : 'cv14_',
            'tags' : ['Swin384', '15Epoch', '5scale', 'LSJ']
         })]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
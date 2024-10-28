_base_ = "./CO_DINO_E15_5scale_Swin_lsj.py"

train_dataloader = dict(
    dataset=dict(
        ann_file='sgkf_5_14/train_2fold.json'
    )
)

val_dataloader = dict(
    dataset=dict(
        ann_file='sgkf_5_14/val_2fold.json'
    )
)

val_evaluator = dict(
    ann_file=_base_.data_root + 'sgkf_5_14/val_2fold.json',
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project' : 'Recycle-Object-Detection',
            'group' : 'CO-DINO-5scale-Swin',
            'name' : 'CO-DINO-5scale-Swin-2fold',
            'entity' : 'cv14_',
            'tags' : ['Swin384', '15Epoch', '5scale', 'LSJ']
         })]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
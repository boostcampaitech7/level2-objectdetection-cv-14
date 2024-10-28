_base_ = "./DINO_E15_5scale_Swin.py"

data_root = "../../../dataset/"

train_dataloader = dict(
    batch_size=1,
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
    ann_file=data_root + 'sgkf_5_14/val_3fold.json',
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
             'project' : 'Recycle-Object-Detection',
             'group' : 'DINO-5scale-Swin',
             'name' : 'Base-DINO-5Scale-Swin-Improve-3fold',
             'entity' : 'cv14_',
             'tags' : ['Swin384', '15Epoch', '5scale']
         })
]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
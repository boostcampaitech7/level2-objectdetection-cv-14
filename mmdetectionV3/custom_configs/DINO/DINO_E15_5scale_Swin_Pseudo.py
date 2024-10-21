_base_ = "./DINO_E15_5scale_Swin.py"

data_root = "../../../dataset/"

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='csv_relabel.json'
    )
)

val_dataloader = None

val_evaluator = None

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=_base_.max_epochs, val_interval=0)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
             'project' : 'Recycle-Object-Detection',
             'group' : 'DINO-5scale-Swin',
             'name' : 'Base-DINO-5Scale-Swin-Improve-Pseudo',
             'entity' : 'cv14_',
             'tags' : ['Swin384', '15Epoch', '5scale', 'Pseudo']
         })
]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
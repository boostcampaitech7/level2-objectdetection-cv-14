# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None) # in boostcamp default : dict(grad_clip=dict(max_keep_ckpts=3, interval=1))  
# learning policy
lr_config = dict(
    policy='CosineAnnealing', 
    by_epoch=False,
    warmup='linear', 
    warmup_iters= 1000, 
    warmup_ratio= 1/10,
    min_lr=1e-07)
runner = dict(type='EpochBasedRunner', max_epochs=12)
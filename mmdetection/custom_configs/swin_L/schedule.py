optimizer_config = dict(grad_clip=None)

optimizer = dict(
    type='AdamW',
    lr=0.00005,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
            }
        )
    )

lr_config = dict(
    policy='step',
    gamma=0.3,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[12, 20])
runner = dict(type='EpochBasedRunner', max_epochs=30)
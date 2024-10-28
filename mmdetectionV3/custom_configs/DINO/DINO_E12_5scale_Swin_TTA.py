_base_ = "./DINO_E12_5scale_Swin.py"

img_scales = [(512,512), (640,640), (768,768), (896, 896), (1024, 1024)]

tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(
                   type='nms',
                   iou_threshold=0.5),
                   max_per_img=100))

tta_pipeline = [
    dict(type='LoadImageFromFile',
         backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(type='Resize', scale=s, keep_ratio=True) for s in img_scales
        ], [
            dict(type='RandomFlip', prob=1.),
            dict(type='RandomFlip', prob=0.)
        ], [
            dict(type='PhotoMetricDistortion')
        ], [
            dict(
               type='PackDetInputs',
               meta_keys=('img_id', 'img_path', 'ori_shape',
                       'img_shape', 'scale_factor', 'flip',
                       'flip_direction'))
       ]])]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
             'project' : 'Recycle-Object-Detection',
             'group' : 'DINO-5scale-Swin',
             'name' : 'Base-DINO-5Scale-Swin-TTA',
             'entity' : 'cv14_',
             'tags' : ['Swin384', '12Epoch', '5scale', 'TTA']
         })
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
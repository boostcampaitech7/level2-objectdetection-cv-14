#!/bin/sh

python train.py /data/ephemeral/home/JYP2/level2-objectdetection-cv-14/mmdetection/custom_configs/ConvNext/cascade_rcnn_ConvNeXt_xlarge_pseudo.py -v
pthon inference.py /data/ephemeral/home/JYP2/level2-objectdetection-cv-14/mmdetection/custom_configs/ConvNext/cascade_rcnn_ConvNeXt_xlarge_pseudo.py

print('------------------------------------------------------------------------------------------------------------------------------------------------')
print('Start Cascade_rcnn_ConvNeXt_xlarge_fold2')

python train.py /data/ephemeral/home/JYP2/level2-objectdetection-cv-14/mmdetection/custom_configs/ConvNext/cascade_rcnn_ConvNeXt_xlarge_fold2.py
python inference.py /data/ephemeral/home/JYP2/level2-objectdetection-cv-14/mmdetection/custom_configs/ConvNext/cascade_rcnn_ConvNeXt_xlarge_fold2.py

print('------------------------------------------------------------------------------------------------------------------------------------------------')
print('Start Cascade_rcnn_ConvNeXt_xlarge_fold3')

python train.py /data/ephemeral/home/JYP2/level2-objectdetection-cv-14/mmdetection/custom_configs/ConvNext/cascade_rcnn_ConvNeXt_xlarge_fold3.py
python inference.py /data/ephemeral/home/JYP2/level2-objectdetection-cv-14/mmdetection/custom_configs/ConvNext/cascade_rcnn_ConvNeXt_xlarge_fold3.py

print('------------------------------------------------------------------------------------------------------------------------------------------------')
print('Start Cascade_rcnn_ConvNeXt_xlarge_fold4')

python train.py /data/ephemeral/home/JYP2/level2-objectdetection-cv-14/mmdetection/custom_configs/ConvNext/cascade_rcnn_ConvNeXt_xlarge_fold4.py
python inference.py /data/ephemeral/home/JYP2/level2-objectdetection-cv-14/mmdetection/custom_configs/ConvNext/cascade_rcnn_ConvNeXt_xlarge_fold4.py

print('------------------------------------------------------------------------------------------------------------------------------------------------')
print('Start Cascade_rcnn_ConvNeXt_xlarge_fold5')

python train.py /data/ephemeral/home/JYP2/level2-objectdetection-cv-14/mmdetection/custom_configs/ConvNext/cascade_rcnn_ConvNeXt_xlarge_fold5.py
python inference.py /data/ephemeral/home/JYP2/level2-objectdetection-cv-14/mmdetection/custom_configs/ConvNext/cascade_rcnn_ConvNeXt_xlarge_fold5.py

print('------------------------------------------------------------------------------------------------------------------------------------------------')
print('End Train & Inference')

'''
NOTE: A few things on effective batch size (EBS), initial learning rate (ILR), and iteration count.
      1. Effective Batch Size (EBS) = (# of GPUs * Batch Size Per GPU)
         EBS is the number of samples processed by your model per iteration across all GPUs. For
         example, if you're training on 2 GPUs and you specify a batch size of 6 in your config 
         file, then your EBS is 12.

      2. New_ILR = Original_ILR * (New_EBS / Original_EBS)
         We manually scale the initial learning rate (ILR) based on the EBS. For example, PIDNet's
         ILR was 0.01 for the Cityscapes Dataset, with an EBS equal to 12. If you had the capacity
         to increase your EBS to 24, then your new ILR would be 0.01*(24/12)=0.02. The same holds
         true if your capacity dictates decreasing EBS to 6, New_ILR=0.01*(6/12)=0.005.

      3. Iteration Count = (# of Training Images / Effective Batch Size) * Original Epochs
         When converting from epoch-based training loops to iteration-based training loops, the
         iteration count is derived from initial total epochs, the number of GPUs, the batch size, 
         and the number of training images. For example, PIDNet's approach trained on the
         Cityscapes train split (2,975 images) for 484 epochs, an EBS of 12 (2 GPUs * 6 Batch Size 
         Per GPU), and an ILR of 0.01. Thus, the total amount of iterations is (2,975/12)*484 =
         119,991 ≈ 120,000 iterations

      We base the Mapillary iteration count on PIDNet's Cityscapes implementation (484 Epochs):
      Mapillary Iteration Count = (18,000 / (2*6)) * 484 = 726,000 iterations
      We also adopt an EBS equal to 12, and an ILR equal to 0.01.      
'''

_base_ = [
    '../_base_/datasets/mapillary_v2.py',
    '../_base_/default_runtime.py'
]
class_weight = [
    1.3348, 1.1980, 1.3508, 0.8861, 0.8473, 0.8291, 0.8951, 1.1403, 0.9788,
    0.9732, 1.0052, 1.0130, 0.8491, 0.8811, 0.8974, 0.9485, 0.9244, 0.9055,
    0.9915, 0.8722, 0.9226, 0.7374, 0.9314, 0.9096, 0.7986, 0.8932, 0.8489,
    0.7498, 1.1196, 0.9468, 0.8868, 1.0377, 0.9937, 1.0203, 1.1937, 0.8897,
    0.8667, 1.1332, 1.1057, 1.0806, 1.0965, 1.0932, 1.1355, 1.1249, 1.0377,
    0.8512, 1.0726, 1.2059, 0.9784, 0.9376, 0.9138, 0.9512, 1.0284, 1.1601,
    0.9656, 1.1798, 1.1069, 1.1829, 1.3145, 0.9029, 1.0519, 0.7235, 0.8884,
    0.8347, 0.7437, 0.9669, 0.9614, 1.0519, 1.0688, 1.0112, 1.1741, 1.1074,
    0.9911, 1.1246, 0.9684, 1.0969, 1.0829, 1.1358, 0.8789, 1.0896, 1.0693,
    1.0230, 1.0719, 0.9085, 0.9673, 0.8471, 1.0491, 0.9459, 0.8756, 1.0594,
    1.2071, 0.9954, 0.9272, 1.0666, 1.1889, 1.1413, 1.0759, 0.9732, 1.0270,
    0.9119, 0.9017, 1.0512, 1.2276, 1.0807, 0.9652, 0.9628, 1.0700, 0.8933,
    0.7926, 1.1705, 0.9659, 1.0125, 0.9812, 1.0628, 0.8799, 0.9404, 1.0990,
    1.2104, 0.9276, 0.9263, 0.8094, 0.8631, 0.8658, 1.0295
    ]
checkpoint_file = "/home/robert.breslin/alessandro/paper_2/mmpretrain/checkpoints/epoch_98.pth"
data_root = '/home/robert.breslin/datasets/mapillary_vistas/'

# preprocessing configuration
crop_size = (512, 1024)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

num_stem_blocks = 3
model = dict(
    type='EncoderDecoderWithFeats',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SEBNet_Staged',          # The type of the backbone module.
        # All fields except `type` come from the __init__ method of class `SEBNet`
        in_channels = 3,
        channels = 64,
        num_stem_blocks = num_stem_blocks,
        num_branch_blocks = 4,
        align_corners = False,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(
        type='DAPPM',
        in_channels=1024, 
        branch_channels=112, 
        out_channels=256, 
        num_scales=5),    # The type of the neck module.
    decode_head=dict(
        type='BaselinePDSBDBASHead',     # The type of the classification head module.
        # All fields except `type` come from the __init__ method of class `LinearClsHead`
        # and you can find them from https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.models.heads.LinearClsHead.html
        num_classes=124,
        in_channels=256,
        num_stem_blocks=num_stem_blocks,
        eval_edges=False,
        loss_decode=[
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0,
                loss_name='loss_seg'),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=0.4,
                loss_name='loss_seg_p'),
            dict(
                type='BoundaryLoss', 
                loss_weight=20.0,
                loss_name='loss_bd'),
            dict(
                type='MultiLabelEdgeLoss',
                loss_weight=5.0,
                loss_name='loss_sbd'),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0,
                loss_name='loss_bas')
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateEdge', edge_width=4),
    dict(type='Mask2Edge', labelIds=list(range(0,124)), radius=2), # 0-19 for cityscapes classes
    dict(type='PackSegInputs')
]
train_dataloader = dict(batch_size=6, dataset=dict(data_root=data_root, pipeline=train_pipeline))
val_dataloader = dict(batch_size=1, dataset=dict(data_root=data_root))
test_dataloader = val_dataloader

iters = 1452000 // 2 # 2 GPUs
val_interval=1000

optim_wrapper = dict(
    # Use SGD optimizer to optimize parameters.
    type='mmpretrain.GradTrackingOptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, 
                   weight_decay=0.0005), clip_grad=None)

# The tuning strategy of the learning rate.
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]

# Training configuration, iterate 100 epochs, and perform validation after every training epoch.
# 'by_epoch=True' means to use `EpochBaseTrainLoop`, 'by_epoch=False' means to use IterBaseTrainLoop.
train_cfg = dict(type='GradientsFeaturesIterTrainLoop', max_iters=iters, val_interval=val_interval)
# Use the default val loop settings.
val_cfg = dict(type='ValLoop')
# Use the default test loop settings.
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, save_begin=iters+1,
        interval=val_interval),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

custom_hooks = [
    dict(
        initial_grads=True,
        interval=iters//10,
        priority='HIGHEST',
        show_plot=False,
        type='mmpretrain.GradFlowVisualizationHook'),
    dict(type='mmpretrain.CustomCheckpointHook', by_epoch=False, interval=-1, 
         save_best=['mAcc', 'mIoU'], rule='greater', save_last=False, priority='VERY_LOW'),
    dict(
        type='FeatureMapVisualizationHook',
        img_name='/home/robert.breslin/datasets/mapillary_vistas/validation/images/_1Gn_xkw7sa_i9GU4mkxxQ.jpg',
        rstrip=None,
        out_dir=None,
        priority='HIGHEST'
    ),
]

randomness = dict(seed=304)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False
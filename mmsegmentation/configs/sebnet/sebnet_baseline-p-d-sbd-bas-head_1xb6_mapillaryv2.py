_base_ = [
    '../_base_/datasets/mapillary_v2.py',
    '../_base_/default_runtime.py'
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
                loss_weight=1.0,
                loss_name='loss_seg'),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
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

iters = 480000
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
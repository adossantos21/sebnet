_base_ = [
    '../_base_/datasets/bdd100k.py',
    '../_base_/default_runtime.py'
]
imagenet_checkpoint_file = "/home/robert.breslin/alessandro/paper_2/mmpretrain/checkpoints/epoch_98.pth"
data_root = "/home/robert.breslin/datasets/bdd100k_seg/bdd100k/seg"

class_weight = [
    0.8238, 0.9277, 0.8431, 1.0055, 0.9627, 0.9685, 1.0660, 1.0259, 0.8433,
    0.9627, 0.8324, 1.0446, 1.2297, 0.8637, 0.9659, 0.9969, 1.2663, 1.2163,
    1.1550
]

# preprocessing configuration
crop_size = (512, 512)
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
        type='SEBNet_Staged', # The type of the backbone module. All fields except `type` come from the __init__ method of class `SEBNet_Staged`
        in_channels = 3,
        channels = 64,
        num_stem_blocks = num_stem_blocks,
        num_branch_blocks = 4,
        align_corners = False,
        init_cfg=dict(type='Pretrained', checkpoint=imagenet_checkpoint_file)),
    neck=dict(
        type='DAPPM', # The type of the neck module.
        in_channels=1024, 
        branch_channels=112, 
        out_channels=256, 
        num_scales=5),    
    decode_head=dict(
        type='Ablation01', # The type of the segmentation decode head module.
        num_classes=19,
        in_channels=256,
        num_stem_blocks=num_stem_blocks,
        eval_edges=False,
        loss_decode=dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0,
                loss_name='loss_seg'),
    ),
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
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=24, 
    dataset=dict(
        data_root=data_root, 
        data_prefix=dict(
            img_path='/home/robert.breslin/datasets/bdd100k_seg/bdd100k/seg/images/train',
            seg_map_path='/home/robert.breslin/datasets/bdd100k_seg/bdd100k/seg/labels/train'
        ),
    pipeline=train_pipeline
    )
)

iters = 87000
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
val_dataloader = dict(
    batch_size=1, 
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(
            img_path='/home/robert.breslin/datasets/bdd100k_seg/bdd100k/seg/images/val',
            seg_map_path='/home/robert.breslin/datasets/bdd100k_seg/bdd100k/seg/labels/val'
        )
    )
)
test_dataloader = val_dataloader

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
        interval=iters // 10,
        priority='HIGHEST',
        show_plot=False,
        type='mmpretrain.GradFlowVisualizationHook'),
    dict(type='mmpretrain.CustomCheckpointHook', by_epoch=False, interval=-1, 
         save_best=['mAcc', 'mIoU'], rule='greater', save_last=False, priority='VERY_LOW'),
    dict(
        type='FeatureMapVisualizationHook',
        img_name='/home/robert.breslin/datasets/bdd100k_seg/bdd100k/seg/labels/train/0a0eaeaf-9ad0c6dd_train_id.png',
        rstrip='_leftImg8bit',
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
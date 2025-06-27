_base_ = [                                    # This config file will inherit all config files in `_base_`.
    '_base_/models/resnet50.py',           # model settings
    '_base_/datasets/imagenet_bs32.py',    # data settings
    '_base_/default_runtime.py'            # runtime settings
]

#work_dir = 'z_test_work_dir'
model = dict(
    type='ImageClassifier',     # The type of the main model (here is for image classification task).
    backbone=dict(
        type='ResNet',          # The type of the backbone module.
        # All fields except `type` come from the __init__ method of class `ResNet`
        # and you can find them from https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.models.backbones.ResNet.html
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=-1,
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),    # The type of the neck module.
    head=dict(
        type='LinearClsHead',     # The type of the classification head module.
        # All fields except `type` come from the __init__ method of class `LinearClsHead`
        # and you can find them from https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.models.heads.LinearClsHead.html
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

dataset_type = 'ImageNet'
# preprocessing configuration
data_preprocessor = dict(
    # Input image data channels in 'RGB' order
    mean=[123.675, 116.28, 103.53],    # Input image normalized channel mean in RGB order
    std=[58.395, 57.12, 57.375],       # Input image normalized channel std in RGB order
    to_rgb=True,                       # Whether to flip the channel from BGR to RGB or RGB to BGR
)

train_pipeline = [
    dict(type='LoadImageFromFile'),     # read image
    dict(type='RandomResizedCrop', scale=224),     # Random scaling and cropping
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),   # random horizontal flip
    dict(type='PackInputs'),         # prepare images and labels
]

test_pipeline = [
    dict(type='LoadImageFromFile'),     # read image
    dict(type='ResizeEdge', scale=256, edge='short'),  # Scale the short side to 256
    dict(type='CenterCrop', crop_size=224),     # center crop
    dict(type='PackInputs'),                 # prepare images and labels
]

# Construct training set dataloader
train_dataloader = dict(
    batch_size=32,                     # batchsize per GPU
    num_workers=5,                     # Number of workers to fetch data per GPU
    dataset=dict(                      # training dataset
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),   # default sampler
    persistent_workers=True,                             # Whether to keep the process, can shorten the preparation time of each epoch
)

# Construct the validation set dataloader
val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)

# The settings of the evaluation metrics for validation. We use the top1 and top5 accuracy here.
val_evaluator = dict(type='Accuracy', topk=(1, 5))

test_dataloader = val_dataloader  # The settings of the dataloader for the test dataset, which is the same as val_dataloader
test_evaluator = val_evaluator    # The settings of the evaluation metrics for test, which is the same as val_evaluator

optim_wrapper = dict(
    # Use SGD optimizer to optimize parameters.
    type='GradTrackingOptimWrapper',
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

# The tuning strategy of the learning rate.
# The 'MultiStepLR' means to use multiple steps policy to schedule the learning rate (LR).
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

# Training configuration, iterate 100 epochs, and perform validation after every training epoch.
# 'by_epoch=True' means to use `EpochBaseTrainLoop`, 'by_epoch=False' means to use IterBaseTrainLoop.
train_cfg = dict(type='GradientTrackingTrainLoop', max_epochs=100, val_interval=1)
#train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
# Use the default val loop settings.
val_cfg = dict()
# Use the default test loop settings.
test_cfg = dict()

# This schedule is for the total batch size 256.
# If you use a different total batch size, like 512 and enable auto learning rate scaling.
# We will scale up the learning rate to 2 times.
auto_scale_lr = dict(base_batch_size=256)

# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, save_begin=74),

    # set sampler seed in a distributed environment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

custom_hooks = [
    dict(type='GradFlowVisualizationHook', interval=10000, initial_grads=True, show_plot=False, priority='NORMAL')
]

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi-process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]  # use local HDD backend
visualizer = dict(
    type='UniversalVisualizer', vis_backends=vis_backends, name='visualizer')

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False
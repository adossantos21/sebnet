_base_ = [
    '../../_base_/models/sebnext_alt.py',
    '../../_base_/datasets/imagenetSubset_bs64_sebnext.py',
    '../../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        arch='baseline',
        drop_path_rate=0.4),
)

# dataset setting
train_dataloader = dict(batch_size=64)

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
# Use the default val loop settings.
val_cfg = dict()
# Use the default test loop settings.
test_cfg = dict()

# runtime setting
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_begin=300)
)

custom_hooks = [
    dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL'),
    dict(type='GradFlowVisualizationHook', interval=300, initial_grads=True, show_plot=False, priority='HIGHEST'),
    dict(type='CustomCheckpointHook', interval=1, save_begin=300, priority='VERY_LOW')
]

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (1 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)

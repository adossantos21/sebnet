# Pre-training with KoLeo Regularization

_base_ = [
    '../../_base_/models/sebnet.py',
    '../../_base_/datasets/imagenetSubset_bs64_sebnet.py',
    '../../_base_/default_runtime.py'
]

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

# This schedule is for the total batch size 256.
# If you use a different total batch size, like 512 and enable auto learning rate scaling.
# We will scale up the learning rate to 2 times.
auto_scale_lr = dict(base_batch_size=64)

# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

# configure default hooks
default_hooks = dict(
    # Override the hook that saves checkpoints per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, save_begin=99),
)

custom_hooks = [
    dict(type='GradFlowVisualizationHook', interval=20000, initial_grads=True, show_plot=False, priority='HIGHEST'),
    dict(type='CustomCheckpointHook', interval=1, save_begin=99, priority='VERY_LOW')
]

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False


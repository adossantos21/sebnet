# Pre-training with KoLeo Regularization

_base_ = [
    '../../_base_/models/sebnet.py',
    '../../_base_/datasets/imagenet_bs64_sebnext.py',
    '../../_base_/schedules/imagenet_bs1024_adamw_sebnext.py',
    '../../_base_/default_runtime.py'
]
model=dict(
    head=dict(
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=None
    )
)

# schedule setting
optim_wrapper = dict(
    type='GradTrackingOptimWrapper',
    optimizer=dict(
        lr=1e-3),
    clip_grad=None,
)


# Training configuration, iterate 100 epochs, and perform validation after every training epoch.
# 'by_epoch=True' means to use `EpochBaseTrainLoop`, 'by_epoch=False' means to use IterBaseTrainLoop.
train_cfg = dict(type='GradientTrackingTrainLoop', max_epochs=300, val_interval=1)

# runtime setting
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_begin=275)
)

custom_hooks = [
    dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL'),
    dict(type='GradFlowVisualizationHook', interval=20000, initial_grads=True, show_plot=False, priority='HIGHEST'),
    dict(type='CustomCheckpointHook', interval=1, save_begin=275, priority='VERY_LOW')
]
# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (1 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
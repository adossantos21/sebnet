_base_ = [
    'pretrain_sebnext.py',
]

model = dict(
    backbone=dict(
        arch='large',
        drop_path_rate=0.4),
)

# schedule setting
optim_wrapper = dict(
    type='GradTrackingOptimWrapper',
    optimizer=dict(
        lr=1e-3),
    clip_grad=None,
)

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False
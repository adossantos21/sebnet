_base_ = [
    'pretrain_sebnext.py',
]

model = dict(
    backbone=dict(
        arch='small',
        drop_path_rate=0.1),
)

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False
_base_ = [
    'pretrain_sebnext.py',
]

model = dict(
    backbone=dict(arch='large'),
)

# load from which checkpoint
load_from = 'path/to/checkpoint.pth'

# whether to resume training from the loaded checkpoint
resume = True
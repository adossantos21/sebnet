_base_ = [
    'pretrain_sebnext.py',
]

model = dict(
    backbone=dict(arch='baseline'),
)

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False


_base_ = [
    'pretrain_sebnext.py',
]

model = dict(
    backbone=dict(
        arch='medium',
        drop_path_rate=0.1),
)

# load from which checkpoint
load_from = '/home/robert.breslin/alessandro/paper_2/mmpretrain/work_dirs/pretrain_sebnext_medium_1xb64_in1k/epoch_27.pth'

# whether to resume training from the loaded checkpoint
resume = True
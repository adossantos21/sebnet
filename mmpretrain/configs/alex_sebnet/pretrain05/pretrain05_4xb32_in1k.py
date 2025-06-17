# Pre-training + Neck with KoLeo Regularization at Neck

_base_ = '../../pretrain02/pretrain02_4xb32_in1k.py'

model = dict(
    neck=dict(
        loss=dict(type='KoLeoLoss', loss_weight=1.0),
    )
)

# trains more epochs
train_cfg = dict(max_epochs=300, val_interval=10)  # Train for 300 epochs, evaluate every 10 epochs
param_scheduler = dict(milestones=[150, 200, 250])   # The learning rate adjustment has also changed

# Use your own dataset directory
train_dataloader = dict(
    dataset=dict(
        data_root='data/imagenet',
        split='train',
        ann_file='meta/train.txt',
        with_label=True),
)
val_dataloader = dict(
    batch_size=64,                  # No back-propagation during validation, larger batch size can be used
    dataset=dict(
        data_root='data/imagenet',
        split='val',
        ann_file='meta/val.txt',
        with_label=True),
)
test_dataloader = dict(
    batch_size=64,                  # No back-propagation during test, larger batch size can be used
    dataset=dict(
        data_root='data/imagenet',
        split='test',
        ann_file='meta/test.txt',
        with_label=True),
)
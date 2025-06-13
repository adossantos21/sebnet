# Pre-training with KoLeo Regularization

_base_ = [
    '../_base_/models/sebnet_neck.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]

# trains more epochs
train_cfg = dict(max_epochs=300, val_interval=10)  # Train for 300 epochs, evaluate every 10 epochs
param_scheduler = dict(step=[150, 200, 250])   # The learning rate adjustment has also changed

# Use your own dataset directory
train_dataloader = dict(
    dataset=dict(data_root='datasets/imagenet/train'),
)
val_dataloader = dict(
    batch_size=64,                  # No back-propagation during validation, larger batch size can be used
    dataset=dict(data_root='datasets/imagenet/val'),
)
test_dataloader = dict(
    batch_size=64,                  # No back-propagation during test, larger batch size can be used
    dataset=dict(data_root='datasets/imagenet/val'),
)
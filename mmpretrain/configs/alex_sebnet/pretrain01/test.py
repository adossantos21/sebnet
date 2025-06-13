_base_ = [
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['configs.pretrain01.pretrain01_4xb32_in1k'],
    allow_failed_imports=False,
)


model = dict(
    type='ImageClassifier',     # The type of the main model (here is for image classification task).
    backbone=dict(
        type='SEBNet',          # The type of the backbone module.
        # All fields except `type` come from the __init__ method of class `SEBNet`
        in_channels = 3,
        channels = 64,
        ppm_channels = 96,
        num_stem_blocks = 2,
        num_branch_blocks = 3,
        align_corners = False),
    head=dict(
        type='DINOHead',     # The type of the classification head module.
        # All fields except `type` come from the __init__ method of class `LinearClsHead`
        # and you can find them from https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.models.heads.LinearClsHead.html
        in_dim=1024,
        out_dim=1000,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

# trains more epochs
train_cfg = dict(max_epochs=300, val_interval=10)  # Train for 300 epochs, evaluate every 10 epochs
param_scheduler = dict(milestones=[150, 200, 250])   # The learning rate adjustment has also changed

# Use your own dataset directory
train_dataloader = dict(
    dataset=dict(data_root='data/imagenet'),
)
val_dataloader = dict(
    batch_size=64,                  # No back-propagation during validation, larger batch size can be used
    dataset=dict(data_root='data/imagenet'),
)
test_dataloader = dict(
    batch_size=64,                  # No back-propagation during test, larger batch size can be used
    dataset=dict(data_root='data/imagenet'),
)
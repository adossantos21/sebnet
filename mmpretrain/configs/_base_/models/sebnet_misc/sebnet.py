model = dict(
    type='ImageClassifier',     # The type of the main model (here is for image classification task).
    backbone=dict(
        type='SEBNet',          # The type of the backbone module.
        # All fields except `type` come from the __init__ method of class `SEBNet`
        in_channels = 3,
        channels = 64,
        ppm_channels = 112,
        num_stem_blocks = 3,
        num_branch_blocks = 4,
        align_corners = False),
    neck=dict(type='GlobalAveragePooling'),    # The type of the neck module.
    head=dict(
        type='SEBNetLinearHead',     # The type of the classification head module.
        # All fields except `type` come from the __init__ method of class `LinearClsHead`
        # and you can find them from https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.models.heads.LinearClsHead.html
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
    init_cfg=dict(
        type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0),
    ]),
)
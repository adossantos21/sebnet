# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='SEBNeXtHigherRes', arch='baseline', drop_path_rate=0.1), # changed to 0.1, 0.5 originally
    neck=dict(type='GlobalAveragePooling'),    # The type of the neck module.
    head=dict(
        type='SEBNetLinearHead',     # The type of the classification head module.
        # All fields except `type` come from the __init__ method of class `LinearClsHead`
        # and you can find them from https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.models.heads.LinearClsHead.html
        num_classes=1000,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=None,
    ),
    init_cfg=dict(
        type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0),
    ]),
)


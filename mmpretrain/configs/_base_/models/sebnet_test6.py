model = dict(
    type='ImageClassifier',     # The type of the main model (here is for image classification task).
    backbone=dict(
        type='SEBNetTest6',          # The type of the backbone module.
        # All fields except `type` come from the __init__ method of class `SEBNet`
        in_channels = 3,
        channels = 64,
        ppm_channels = 96,
        num_stem_blocks = 2,
        num_branch_blocks = 3,
        align_corners = False),
    neck=dict(type='GlobalAveragePooling'),    # The type of the neck module.
    head=dict(
        type='FCNHead',     # The type of the classification head module.
        # All fields except `type` come from the __init__ method of class `LinearClsHead`
        # and you can find them from https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.models.heads.LinearClsHead.html
        num_classes=1000,
        in_channels_main=2048,
        in_channels_aux=1024,
        loss1=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss2=dict(type='CrossEntropyLoss', loss_weight=1.0)
    ))
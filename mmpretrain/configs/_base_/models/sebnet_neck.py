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
    neck=dict(
        type='DAPPM',
        in_channels=2048,
        branch_channels=96,
        out_channels=256,
        num_scales=5,
        kernel_sizes = [5, 9, 17],
        strides = [2, 4, 8],
        paddings = [2, 4, 8],
        norm_cfg = dict(type='BN', momentum=0.1),
        act_cfg = dict(type='ReLU', inplace=True),
        conv_cfg = dict(
            order=('norm', 'act', 'conv'), bias=False),
        upsample_mode = 'bilinear',
    ),
    head=dict(
        type='SEBNetLinearHead',     # The type of the classification head module.
        # All fields except `type` come from the __init__ method of class `LinearClsHead`
        # and you can find them from https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.models.heads.LinearClsHead.html
        num_classes=1000,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
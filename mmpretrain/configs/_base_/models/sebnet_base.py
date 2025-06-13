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
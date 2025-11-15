_base_ = './sebnet_1xb24-87k_bdd100k.py'
class_weight = [
    0.8238, 0.9277, 0.8431, 1.0055, 0.9627, 0.9685, 1.0660, 1.0259, 0.8433,
    0.9627, 0.8324, 1.0446, 1.2297, 0.8637, 0.9659, 0.9969, 1.2663, 1.2163,
    1.1550
]
crop_size = (512, 512)
model = dict(
    backbone=dict(
        init_cfg=None,
    ),
    decode_head=dict(
        type='Ablation20',
        loss_decode=[
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0,
                loss_name='loss_seg'),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=0.4,
                loss_name='loss_seg_p'),
            dict(
                type='BoundaryLoss', 
                loss_weight=20.0,
                loss_name='loss_hed'),
            dict(
                type='MultiLabelEdgeLoss',
                loss_weight=5.0,
                loss_name='loss_sbd'),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0,
                loss_name='loss_bas')
        ]
    )
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(1280, 720),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateEdge', edge_width=4),
    dict(type='Mask2Edge', labelIds=list(range(0,19)), radius=2), # 0-19 for cityscapes classes
    dict(type='PackSegInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

load_from = '/home/robert.breslin/alessandro/testing/paper_2/mmsegmentation/work_dirs/sebnet_baseline-p-d-sbd-bas-head_2xb6_mapillaryv2/20250916_135803/checkpoints/remapped_checkpoint.pth'
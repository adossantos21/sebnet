_base_ = './sebnet_1xb6-240k_cityscapes.py'
class_weight = [
    0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786,
    1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
    1.0507
]
crop_size = (1024, 1024)
model = dict(
    decode_head=dict(
        type='Ablation34',
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
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Mask2Edge', labelIds=list(range(0,19)), radius=2), # 0-19 for cityscapes classes
    dict(type='PackSegInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
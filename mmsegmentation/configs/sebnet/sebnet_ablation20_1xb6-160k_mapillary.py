_base_ = './sebnet_2xb6-726k_mapillary.py'
class_weight = [
    1.3348, 1.1980, 1.3508, 0.8861, 0.8473, 0.8291, 0.8951, 1.1403, 0.9788,
    0.9732, 1.0052, 1.0130, 0.8491, 0.8811, 0.8974, 0.9485, 0.9244, 0.9055,
    0.9915, 0.8722, 0.9226, 0.7374, 0.9314, 0.9096, 0.7986, 0.8932, 0.8489,
    0.7498, 1.1196, 0.9468, 0.8868, 1.0377, 0.9937, 1.0203, 1.1937, 0.8897,
    0.8667, 1.1332, 1.1057, 1.0806, 1.0965, 1.0932, 1.1355, 1.1249, 1.0377,
    0.8512, 1.0726, 1.2059, 0.9784, 0.9376, 0.9138, 0.9512, 1.0284, 1.1601,
    0.9656, 1.1798, 1.1069, 1.1829, 1.3145, 0.9029, 1.0519, 0.7235, 0.8884,
    0.8347, 0.7437, 0.9669, 0.9614, 1.0519, 1.0688, 1.0112, 1.1741, 1.1074,
    0.9911, 1.1246, 0.9684, 1.0969, 1.0829, 1.1358, 0.8789, 1.0896, 1.0693,
    1.0230, 1.0719, 0.9085, 0.9673, 0.8471, 1.0491, 0.9459, 0.8756, 1.0594,
    1.2071, 0.9954, 0.9272, 1.0666, 1.1889, 1.1413, 1.0759, 0.9732, 1.0270,
    0.9119, 0.9017, 1.0512, 1.2276, 1.0807, 0.9652, 0.9628, 1.0700, 0.8933,
    0.7926, 1.1705, 0.9659, 1.0125, 0.9812, 1.0628, 0.8799, 0.9404, 1.0990,
    1.2104, 0.9276, 0.9263, 0.8094, 0.8631, 0.8658, 1.0295
    ]
crop_size = (512, 1024)
model = dict(
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
        scale=(2048, 1024),
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
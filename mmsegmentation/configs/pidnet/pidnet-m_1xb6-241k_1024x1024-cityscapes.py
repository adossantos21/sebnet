_base_ = './pidnet-s_2xb6-120k_1024x1024-cityscapes.py'
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-m_imagenet1k_20230306-39893c52.pth'  # noqa
model = dict(
    backbone=dict(channels=64, init_cfg=dict(checkpoint=checkpoint_file)),
    decode_head=dict(in_channels=256, act_cfg=dict(type='ReLU', inplace=False)))

iters = 241000
val_interval = 1000

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

train_cfg = dict(type='IterBasedTrainLoop', max_iters=iters, val_interval=val_interval)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, save_begin=iters+1,
        interval=val_interval),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

custom_hooks = [
    dict(type='mmpretrain.CustomCheckpointHook', by_epoch=False, interval=-1, 
         save_best=['mAcc', 'mIoU'], rule='greater', save_last=False, priority='VERY_LOW'),
    dict(
        type='FeatureMapVisualizationHook',
        img_name='data/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png',
        rstrip='_leftImg8bit',
        out_dir=None,
        priority='HIGHEST'
    ),
]
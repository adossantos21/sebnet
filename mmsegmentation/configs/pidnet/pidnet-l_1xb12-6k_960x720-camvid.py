_base_ = './pidnet-s_1xb12-6k_960x720-camvid.py'
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-l_imagenet1k_20230306-67889109.pth'  # noqa
model = dict(
    backbone=dict(
        channels=64,
        ppm_channels=112,
        num_stem_blocks=3,
        num_branch_blocks=4,
        init_cfg=dict(checkpoint=checkpoint_file)),
    decode_head=dict(in_channels=256, channels=256))

iters = 6200 * 2 # 6200 original
val_interval = 50

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
]
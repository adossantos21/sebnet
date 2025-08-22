# python tools/test.py path/to/your_config.py path/to/your_checkpoint.pth --eval mIoU
python tools/test.py \
    configs/sebnet/sebnet_baseline-head_1xb6_cityscapes_ce.py \
    work_dirs/sebnet_baseline-head_1xb6_cityscapes_ce/iter_120000.pth
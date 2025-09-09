# python tools/test.py path/to/your_config.py path/to/your_checkpoint.pth --eval mIoU
export CUDA_VISIBLE_DEVICES=0
python tools/test.py \
    configs/sebnet/sebnet_baseline-p-d-bas-head_1xb6_cityscapes.py \
    work_dirs/sebnet_baseline-p-d-bas-head_1xb6_cityscapes/20250906_105242/checkpoints/sebnet_baseline-p-d-bas-head_1xb6_cityscapes/20250906_105242/best_mIoU.pth
# python tools/test.py path/to/your_config.py path/to/your_checkpoint.pth --eval mIoU
export CUDA_VISIBLE_DEVICES=3
python tools/test.py \
    configs/sebnet/sebnet_ablation40_1xb6-160k_cityscapes.py \
    /home/robert.breslin/alessandro/testing/paper_2/mmsegmentation/work_dirs/sebnet_ablation40_1xb6-160k_cityscapes-testset/iter_276000.pth \

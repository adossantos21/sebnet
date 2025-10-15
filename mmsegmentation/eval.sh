# python tools/test.py path/to/your_config.py path/to/your_checkpoint.pth --eval mIoU
export CUDA_VISIBLE_DEVICES=2
python tools/test.py \
    /home/robert.breslin/alessandro/testing/paper_2/mmsegmentation/configs/sebnet/sebnet_ablation40_1xb6-160k_mapillary.py \
    /home/robert.breslin/alessandro/testing/paper_2/mmsegmentation/work_dirs/sebnet_baseline-p-d-sbd-bas-head_2xb6_mapillaryv2/20250916_135803/checkpoints/remapped_checkpoint.pth
    #/home/robert.breslin/alessandro/testing/paper_2/mmsegmentation/work_dirs/sebnet_baseline-p-sbd-bas-head-conditioned_2xb6_mapillaryv2/20250916_155205/checkpoints/remapped_checkpoint.pth
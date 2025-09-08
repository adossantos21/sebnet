export CUDA_VISIBLE_DEVICES=0
python mmsegmentation/tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-p-sbd-bas-head-conditioned_1xb6_cityscapes.py \
    mmsegmentation/work_dirs/sebnet_baseline-p-sbd-bas-head-conditioned_1xb6_cityscapes/20250903_224209/checkpoints/sebnet_baseline-p-sbd-bas-head-conditioned_1xb6_cityscapes/20250903_224209/best_mIoU.pth \
    mmsegmentation/work_dirs/sebnet_baseline-d-multilabel-head_1xb6_cityscapes/20250903_224209/SBD_preds/

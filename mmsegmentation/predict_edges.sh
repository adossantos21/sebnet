export CUDA_VISIBLE_DEVICES=0
python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-d-multilabel-head_1xb6_cityscapes.py \
    work_dirs/sebnet_baseline-d-multilabel-head_1xb6_cityscapes/20250826_134843/checkpoints/sebnet_baseline-d-multilabel-head_1xb6_cityscapes/20250826_134843/best_mIoU.pth \
    work_dirs/sebnet_baseline-d-multilabel-head_1xb6_cityscapes/20250826_134843/SBD_preds/

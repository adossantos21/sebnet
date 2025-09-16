export CUDA_VISIBLE_DEVICES=0
python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-p-d-bas-head_1xb6_cityscapes.py \
    SBD/ablation25/ckpt/160K/best_mIoU.pth \
    SBD/ablation25/preds/160K/

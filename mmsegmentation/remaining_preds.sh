export CUDA_VISIBLE_DEVICES=1
python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-d-head-earlier-layers_1xb6_cityscapes.py \
    SBD/ablation08/ckpt/best_mIoU.pth \
    SBD/ablation08/preds/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-d-multilabel-head-earlier-layers_1xb6_cityscapes.py \
    SBD/ablation12/ckpt/best_mIoU.pth \
    SBD/ablation12/preds/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-p-d-head_1xb6_cityscapes.py \
    SBD/ablation19/ckpt/best_mIoU.pth \
    SBD/ablation19/preds/
export CUDA_VISIBLE_DEVICES=0
python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-d-multilabel-head_1xb6_cityscapes.py \
    SBD/ablation07/ckpt/best_mIoU.pth \
    SBD/ablation07/preds/    

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-d-head-earlier-layers_1xb6_cityscapes.py \
    SBD/ablation08/ckpt/best_mIoU.pth \
    SBD/ablation08/preds/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-casenet-head-earlier-layers_1xb6_cityscapes.py \
    SBD/ablation09/ckpt/best_mIoU.pth \
    SBD/ablation09/preds/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-dff-head-earlier-layers_1xb6_cityscapes.py \
    SBD/ablation10/ckpt/best_mIoU.pth \
    SBD/ablation10/preds/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-d-multilabel-head-earlier-layers_1xb6_cityscapes.py \
    SBD/ablation12/ckpt/best_mIoU.pth \
    SBD/ablation12/preds/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-d-multilabel-head_1xb6_cityscapes.py \
    SBD/ablation13/ckpt/best_mIoU.pth \
    SBD/ablation13/preds/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-d-multilabel-head_1xb6_cityscapes.py \
    SBD/ablation14/ckpt/best_mIoU.pth \
    SBD/ablation14/preds/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-d-multilabel-head_1xb6_cityscapes.py \
    SBD/ablation15/ckpt/best_mIoU.pth \
    SBD/ablation15/preds/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-d-multilabel-head_1xb6_cityscapes.py \
    SBD/ablation16/ckpt/best_mIoU.pth \
    SBD/ablation16/preds/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-d-multilabel-head_1xb6_cityscapes.py \
    SBD/ablation17/ckpt/best_mIoU.pth \
    SBD/ablation17/preds/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-d-multilabel-head_1xb6_cityscapes.py \
    SBD/ablation18/ckpt/best_mIoU.pth \
    SBD/ablation18/preds/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-p-d-head_1xb6_cityscapes.py \
    SBD/ablation19/ckpt/best_mIoU.pth \
    SBD/ablation19/preds/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-d-sbd-head_1xb6_cityscapes.py \
    SBD/ablation20/ckpt/best_mIoU.pth \
    SBD/ablation20/preds/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-p-sbd-head-conditioned_1xb6_cityscapes.py \
    SBD/ablation21/ckpt/best_mIoU.pth \
    SBD/ablation21/preds/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-p-sbd-head-fused_1xb6_cityscapes.py \
    SBD/ablation22/ckpt/best_mIoU.pth \
    SBD/ablation22/preds/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-p-sbd-bas-head-conditioned_1xb6_cityscapes.py \
    SBD/ablation23/ckpt/160K/best_mIoU.pth \
    SBD/ablation23/preds/160K/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-p-sbd-bas-head-conditioned_1xb6_cityscapes.py \
    SBD/ablation23/ckpt/240K/best_mIoU.pth \
    SBD/ablation23/preds/240K/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-p-sbd-bas-head-fused_1xb6_cityscapes.py \
    SBD/ablation24/ckpt/160K/best_mIoU.pth \
    SBD/ablation24/preds/160K/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-p-sbd-bas-head-fused_1xb6_cityscapes.py \
    SBD/ablation24/ckpt/240K/best_mIoU.pth \
    SBD/ablation24/preds/240K/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-p-d-bas-head_1xb6_cityscapes.py \
    SBD/ablation25/ckpt/160K/best_mIoU.pth \
    SBD/ablation25/preds/160K/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-p-d-bas-head_1xb6_cityscapes.py \
    SBD/ablation25/ckpt/240K/best_mIoU.pth \
    SBD/ablation25/preds/240K/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-p-d-sbd-head_1xb6_cityscapes.py \
    SBD/ablation26/ckpt/best_mIoU.pth \
    SBD/ablation26/preds/   

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-p-d-sbd-bas-head_1xb6_cityscapes.py \
    SBD/ablation27/ckpt/160K/best_mIoU.pth \
    SBD/ablation27/preds/160K/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-p-d-sbd-bas-head_1xb6_cityscapes.py \
    SBD/ablation27/ckpt/240K/best_mIoU.pth \
    SBD/ablation27/preds/240K/  

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-bem-head-earlier-layers_1xb6_cityscapes.py \
    SBD/ablation11/ckpt/best_mIoU.pth \
    SBD/ablation11/preds/

python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-bem-head_1xb6_cityscapes.py \
    SBD/ablation06/ckpt/best_mIoU.pth \
    SBD/ablation06/preds/
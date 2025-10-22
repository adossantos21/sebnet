export CUDA_VISIBLE_DEVICES=3
python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_baseline-p-sbd-bas-head-conditioned_1xb6_cityscapes.py \
    /home/robert.breslin/alessandro/testing/paper_2/mmsegmentation/work_dirs/sebnet_ablation33_1xb6-160k_cityscapes/20250906_102650/checkpoints/best_mIoU.pth \
    SBD/table5Best/preds/240K/

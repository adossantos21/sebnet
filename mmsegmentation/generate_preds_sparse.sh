export CUDA_VISIBLE_DEVICES=2
python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_ablation40_1xb6-160k_cityscapes.py \
    /home/robert.breslin/alessandro/testing/paper_2/mmsegmentation/work_dirs/sebnet_ablation40_1xb6-160k_cityscapes/20251013_134001/checkpoints/best_mIoU.pth \
    SBD/table6Best_Mapillary/preds/240K/

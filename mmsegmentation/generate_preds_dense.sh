export CUDA_VISIBLE_DEVICES=3
python tools/analysis_tools/generate_dense_predictions.py \
    /home/robert.breslin/alessandro/testing/paper_2/mmsegmentation/configs/sebnet/sebnet_ablation33_1xb6-160k_cityscapes.py \
    /home/robert.breslin/alessandro/testing/paper_2/mmsegmentation/work_dirs/sebnet_ablation33_1xb6-160k_cityscapes/20251013_133840/checkpoints/best_mIoU.pth \
    preds/dense/table3BestWithMapillary/

python tools/analysis_tools/generate_dense_predictions.py \
    /home/robert.breslin/alessandro/testing/paper_2/mmsegmentation/configs/sebnet/sebnet_ablation33_1xb6-160k_cityscapes.py \
    /home/robert.breslin/alessandro/testing/paper_2/mmsegmentation/work_dirs/sebnet_ablation33_1xb6-160k_cityscapes-testset/iter_281000.pth \
    preds/dense/table3BestWithMapillaryAndTestSet/

python tools/analysis_tools/generate_dense_predictions.py \
    configs/sebnet/sebnet_ablation40_1xb6-160k_cityscapes.py \
    /home/robert.breslin/alessandro/testing/paper_2/mmsegmentation/work_dirs/sebnet_ablation40_1xb6-160k_cityscapes/20251013_134001/checkpoints/best_mIoU.pth \
    preds/dense/table4BestWithMapillary/

python tools/analysis_tools/generate_dense_predictions.py \
    configs/sebnet/sebnet_ablation40_1xb6-160k_cityscapes.py \
    /home/robert.breslin/alessandro/testing/paper_2/mmsegmentation/work_dirs/sebnet_ablation40_1xb6-160k_cityscapes-testset/iter_281000.pth \
    preds/dense/table4BestWithMapillaryAndTestSet/

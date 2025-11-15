export CUDA_VISIBLE_DEVICES=0
python tools/analysis_tools/generate_edge_predictions.py \
    configs/sebnet/sebnet_ablation49_1xb6-160k_cityscapes.py \
    path/to/checkpoint.pth \
    output/path/
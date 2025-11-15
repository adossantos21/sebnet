export CUDA_VISIBLE_DEVICES=0

python tools/analysis_tools/generate_dense_predictions.py \
    configs/pidnet/pidnet-l_1xb12-6k_960x720-camvid.py \
    path/to/checkpoint.pth \
    output/path/ \
    val # split
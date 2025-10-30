export CUDA_VISIBLE_DEVICES=2
python tools/analysis_tools/get_flops.py \
    configs/stdc/stdc2_in1k-pre_4xb12-80k_cityscapes-512x1024.py \
    --shape 2048 1024